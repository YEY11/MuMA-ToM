"""
Perception Pipeline
Orchestrates visual extraction from poker video frames
"""

import json
import os
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

from LIMP_Poker_V3.config import config
from LIMP_Poker_V3.core.registry import AgentRegistry
from LIMP_Poker_V3.core.schema import EpisodeData, GameState, PhaseData, PhaseType

# Import agents package to trigger registration decorators
from LIMP_Poker_V3.perception import agents as _perception_agents  # noqa: F401
from LIMP_Poker_V3.preprocessing.video_preprocessor import get_frame_paths


class PerceptionPipeline:
    """
    Main perception pipeline that coordinates:
    1. Frame-by-frame VLM extraction (BoardAgent)
    2. Action detection from state transitions (ActionDetector)
    3. Phase segmentation and timeline construction
    """

    def __init__(self):
        # Get enabled perception agents
        self.agents = AgentRegistry.get_perception_agents(config.AGENT_CONFIG)
        logger.info(
            f"Initialized PerceptionPipeline with agents: {[a.name for a in self.agents]}"
        )

        # Find specific agents
        self.board_agent = None
        self.action_detector = None
        for agent in self.agents:
            if agent.name == "BoardAgent":
                self.board_agent = agent
            elif agent.name == "ActionDetector":
                self.action_detector = agent

    def run(
        self,
        episode_dir: str,
        output_path: Optional[str] = None,
        use_cache: bool = True,
    ) -> EpisodeData:
        """
        Run the perception pipeline on an episode.

        Args:
            episode_dir: Directory containing frames/ and audio.wav
            output_path: Path to save output JSON (optional)
            use_cache: Whether to use cached raw states if available

        Returns:
            EpisodeData with extracted perception results
        """
        episode_id = os.path.basename(episode_dir)
        frames_dir = os.path.join(episode_dir, "frames")
        cache_path = os.path.join(episode_dir, "raw_states_cache.json")

        logger.info(f"Starting perception pipeline for {episode_id}")

        if not os.path.exists(frames_dir):
            logger.error(f"Frames directory not found: {frames_dir}")
            return EpisodeData(
                episode_id=episode_id,
                protocol=config.PROTOCOL_MODE,
            )

        # 1. Extract raw states from frames
        raw_states = self._extract_raw_states(frames_dir, cache_path, use_cache)

        if not raw_states:
            logger.error("No states extracted")
            return EpisodeData(
                episode_id=episode_id,
                protocol=config.PROTOCOL_MODE,
            )

        # 2. Build timeline with phases and actions
        timeline = self._build_timeline(raw_states)

        # 3. Construct episode data
        episode_data = EpisodeData(
            episode_id=episode_id,
            protocol=config.PROTOCOL_MODE,
            meta={
                "fps": config.FPS,
                "frame_count": len(raw_states),
                "duration": raw_states[-1].timestamp if raw_states else 0,
            },
            timeline=timeline,
        )

        # 4. Save output
        if output_path:
            with open(output_path, "w") as f:
                f.write(episode_data.model_dump_json(indent=2))
            logger.info(f"Perception output saved to {output_path}")

        return episode_data

    def _extract_raw_states(
        self,
        frames_dir: str,
        cache_path: str,
        use_cache: bool,
    ) -> List[GameState]:
        """
        Extract game states from frames using VLM.

        Args:
            frames_dir: Directory containing frame images
            cache_path: Path to cache file
            use_cache: Whether to use cached results

        Returns:
            List of GameState objects
        """
        # Try loading from cache
        if use_cache and os.path.exists(cache_path):
            logger.info("Loading raw states from cache...")
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                states = [GameState(**s) for s in cached_data]
                logger.info(f"Loaded {len(states)} cached states")
                return states
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, re-extracting...")

        if not self.board_agent:
            logger.error("BoardAgent not available")
            return []

        # Get frame paths
        frames = get_frame_paths(frames_dir)
        if not frames:
            logger.error("No frames found")
            return []

        # Sampling step
        fps = config.FPS
        step = max(1, int(fps * config.SAMPLING_INTERVAL))

        logger.info(f"Extracting states (frames: {len(frames)}, step: {step})")

        raw_states = []
        for i in tqdm(range(0, len(frames), step), desc="Extracting"):
            frame_path = frames[i]
            timestamp = i / fps

            # Extract using VLM
            raw_data = self.board_agent.process(frame_path, timestamp)

            # Convert to GameState
            state = self.board_agent.parse_to_game_state(raw_data, timestamp)
            raw_states.append(state)

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump([s.model_dump() for s in raw_states], f, indent=2)
            logger.info(f"Saved {len(raw_states)} states to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

        return raw_states

    # Phase order for validation (phases can only progress forward)
    PHASE_ORDER = {
        PhaseType.PRE_FLOP: 0,
        PhaseType.FLOP: 1,
        PhaseType.TURN: 2,
        PhaseType.RIVER: 3,
        PhaseType.SHOWDOWN: 4,
        PhaseType.UNKNOWN: -1,
    }

    def _stabilize_phases(
        self, states: List[GameState], min_consecutive: int = 3
    ) -> List[GameState]:
        """
        Stabilize phase detection by applying debouncing logic.

        Rules:
        1. Phases can only progress forward ONE step at a time:
           Pre-flop → Flop → Turn → River → Showdown
        2. Unknown/transition frames inherit previous valid phase
        3. Require min_consecutive frames to confirm phase change
        4. Reject phase jumps (e.g., Flop → River skips Turn)

        Args:
            states: Raw game states from VLM
            min_consecutive: Minimum consecutive frames needed to confirm phase change

        Returns:
            States with stabilized phases
        """
        if not states:
            return states

        # Find first valid (non-Unknown) phase
        current_phase = PhaseType.PRE_FLOP
        for state in states:
            if state.phase != PhaseType.UNKNOWN:
                current_phase = state.phase
                break

        stabilized = []
        pending_phase = None
        pending_count = 0

        for state in states:
            detected_phase = state.phase

            # Skip Unknown phases - inherit current phase
            if detected_phase == PhaseType.UNKNOWN:
                state.phase = current_phase
                stabilized.append(state)
                continue

            # Check if this is a valid forward transition
            detected_order = self.PHASE_ORDER.get(detected_phase, -1)
            current_order = self.PHASE_ORDER.get(current_phase, -1)
            order_diff = detected_order - current_order

            if order_diff == 1:
                # Valid single-step forward transition (e.g., Flop → Turn)
                if pending_phase == detected_phase:
                    pending_count += 1
                else:
                    pending_phase = detected_phase
                    pending_count = 1

                # Confirm transition after min_consecutive frames
                if pending_count >= min_consecutive:
                    logger.debug(
                        f"Phase transition confirmed: {current_phase} → {detected_phase} "
                        f"at {state.timestamp}s"
                    )
                    current_phase = detected_phase
                    pending_phase = None
                    pending_count = 0

            elif order_diff > 1:
                # Invalid jump transition (e.g., Flop → River skips Turn) - ignore
                logger.debug(
                    f"Ignoring phase jump: {current_phase} → {detected_phase} "
                    f"at {state.timestamp}s (skips {order_diff - 1} phase(s))"
                )
                pending_phase = None
                pending_count = 0

            elif order_diff < 0:
                # Invalid backward transition - ignore and use current phase
                logger.debug(
                    f"Ignoring backward transition: {current_phase} → {detected_phase} "
                    f"at {state.timestamp}s"
                )
                pending_phase = None
                pending_count = 0

            else:
                # Same phase - reset pending
                pending_phase = None
                pending_count = 0

            # Apply stabilized phase
            state.phase = current_phase
            stabilized.append(state)

        logger.info(f"Phase stabilization complete: {len(stabilized)} states processed")
        return stabilized

    def _build_timeline(self, states: List[GameState]) -> List[PhaseData]:
        """
        Build timeline by segmenting phases and detecting actions.

        Args:
            states: List of GameState objects

        Returns:
            List of PhaseData objects
        """
        if not states:
            return []

        # Apply phase stabilization
        states = self._stabilize_phases(states)

        timeline = []
        current_phase_states = []
        current_phase = states[0].phase
        phase_start_time = states[0].timestamp

        for state in states:
            # Check for phase change
            if state.phase != current_phase and state.phase != PhaseType.UNKNOWN:
                # Close current phase
                if current_phase_states:
                    phase_data = self._create_phase_data(
                        current_phase,
                        phase_start_time,
                        current_phase_states,
                    )
                    timeline.append(phase_data)

                # Start new phase
                current_phase = state.phase
                phase_start_time = state.timestamp
                current_phase_states = [state]
            else:
                current_phase_states.append(state)

        # Close final phase
        if current_phase_states:
            phase_data = self._create_phase_data(
                current_phase,
                phase_start_time,
                current_phase_states,
            )
            timeline.append(phase_data)

        logger.info(f"Timeline built with {len(timeline)} phases")
        return timeline

    def _create_phase_data(
        self,
        phase: PhaseType,
        start_time: float,
        states: List[GameState],
    ) -> PhaseData:
        """
        Create PhaseData with detected actions.

        Args:
            phase: Phase type
            start_time: Phase start time
            states: States within this phase

        Returns:
            PhaseData object
        """
        actions = []

        if self.action_detector and len(states) >= 2:
            # Detect actions from state transitions
            for i in range(1, len(states)):
                prev_state = states[i - 1]
                curr_state = states[i]

                # Get interval states for behavioral analysis
                interval_states = states[max(0, i - 5) : i + 1]

                detected = self.action_detector.detect_actions(
                    prev_state, curr_state, interval_states
                )

                for action in detected:
                    logger.debug(
                        f"Action: {action.player_name} {action.action_type} "
                        f"${action.amount} at {action.timestamp}s"
                    )
                    actions.append(action)

        return PhaseData(
            phase=phase,
            start_time=start_time,
            end_time=states[-1].timestamp,
            actions=actions,
            initial_state=states[0],
            final_state=states[-1],
        )


def run_perception(
    video_path: str,
    output_dir: str,
    skip_preprocess: bool = False,
) -> EpisodeData:
    """
    Convenience function to run preprocessing and perception.

    Args:
        video_path: Path to input video
        output_dir: Output directory for episode
        skip_preprocess: Skip video preprocessing if already done

    Returns:
        EpisodeData with perception results
    """
    from LIMP_Poker_V3.preprocessing.video_preprocessor import preprocess_video

    episode_id = os.path.splitext(os.path.basename(video_path))[0]
    episode_dir = os.path.join(output_dir, episode_id)

    # Preprocess
    if not skip_preprocess:
        result = preprocess_video(video_path, episode_dir)
        if not result.get("success"):
            logger.error(f"Preprocessing failed: {result.get('error')}")
            return EpisodeData(episode_id=episode_id, protocol=config.PROTOCOL_MODE)

    # Run perception
    pipeline = PerceptionPipeline()
    output_path = os.path.join(episode_dir, "perception_output.json")

    return pipeline.run(episode_dir, output_path)
