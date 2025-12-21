import os
import json
import glob
from loguru import logger
from tqdm import tqdm
from LIMP_Poker_V2.config import config
from LIMP_Poker_V2.core.schema import (
    EpisodeData,
    PhaseData,
    ActionEvent,
    GameState,
)
from LIMP_Poker_V2.perception.agents.action_agent import ActionDetectionAgent
from LIMP_Poker_V2.perception.agents.board_agent import BoardParsingAgent
from LIMP_Poker_V2.annotation.audio_gt_agent import AudioGTAgent


class PerceptionPipeline:
    def __init__(self):
        # Only BoardAgent (Perception) and ActionAgent (Logic) remain
        self.board_agent = BoardParsingAgent()
        self.action_agent = ActionDetectionAgent()
        self.audio_gt_agent = AudioGTAgent()

    def run(self, episode_dir: str, output_path: str):
        logger.info(f"Starting perception pipeline for {episode_dir}")

        frames_dir = os.path.join(episode_dir, "frames")
        if not os.path.exists(frames_dir):
            logger.error(f"Frames directory not found: {frames_dir}")
            return

        # --- ONE-PASS EXTRACTION ---
        # Instead of scanning for Phases then Actions, we scan once.
        # We cache the raw LLM outputs to allow re-running logic without re-spending tokens.

        frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not frames:
            logger.error("No frames found")
            return

        # Sampling Step (Assuming 1s interval is desired, and FPS=1 from config)
        # If FPS=1, then every frame is 1s apart. We scan all.
        # If FPS=5, we might scan every 5 frames for 1s interval, or scan all for finer grain.
        # Let's use config.SAMPLING_INTERVAL to decide.
        fps = config.FPS
        step = max(1, int(fps * config.SAMPLING_INTERVAL))

        raw_states = []
        cache_path = os.path.join(episode_dir, "raw_states_cache.json")

        if os.path.exists(cache_path):
            logger.info("Loading raw states from cache...")
            with open(cache_path, "r") as f:
                raw_data = json.load(f)
                # Reconstruct GameState objects
                for d in raw_data:
                    # Parse players back to objects if needed, or rely on dict compatibility
                    # For robust reconstruction we should map back to Schema
                    # But for now let's assume we re-parse or use dicts
                    pass
                # Ideally we reload them into Pydantic models.
                # For simplicity in this diff, we assume we re-run if cache logic isn't fully implemented
                # Or we just skip extraction loop.
                # Let's Re-Run for now to ensure we use the new Schema

        logger.info(
            f"Starting One-Pass Extraction (Total frames: {len(frames)}, Step: {step})..."
        )

        for i in tqdm(range(0, len(frames), step), desc="Extracting"):
            frame_path = frames[i]
            # Use BoardAgent to extract EVERYTHING (Phase, Action info, Micro-gestures)
            # Timestamp is approx i / fps
            ts = i / fps
            state = self.board_agent.parse_state(frame_path, timestamp=ts)
            raw_states.append(state)

        # Save Cache
        with open(cache_path, "w") as f:
            # Dump Pydantic models
            json.dump([s.model_dump() for s in raw_states], f, indent=2)

        # --- LOGIC PROCESSING (Offline) ---
        # Now we process the raw_states to generate Phases and Actions
        # without calling LLM again.

        timeline = self._process_raw_states(raw_states)

        # 3. Ground Truth (Annotation)
        audio_path = os.path.join(episode_dir, "audio.wav")
        gt_data = {}
        if os.path.exists(audio_path):
            logger.info("Extracting Ground Truth from Audio...")
            gt_data = self.audio_gt_agent.transcribe_and_extract(audio_path)

        # 4. Construct Final Episode Data
        episode = EpisodeData(
            episode_id=os.path.basename(episode_dir),
            meta={"fps": fps},
            timeline=timeline,
            ground_truth=gt_data,
        )

        # 5. Save
        with open(output_path, "w") as f:
            f.write(episode.model_dump_json(indent=2))

        logger.info(f"Pipeline finished. Output saved to {output_path}")

    def _process_raw_states(self, raw_states: list[GameState]) -> list[PhaseData]:
        """
        Reconstruct Phases and Actions from the sequence of GameStates.
        """
        timeline = []
        if not raw_states:
            return []

        current_phase_states = []
        current_phase_type = raw_states[0].phase
        phase_start_time = raw_states[0].timestamp

        last_state = raw_states[0]

        for i, state in enumerate(raw_states):
            # 1. Check Phase Change
            if state.phase != current_phase_type:
                # Phase Ended
                # Process Actions within this phase
                actions = self._detect_actions_in_sequence(current_phase_states)

                phase_data = PhaseData(
                    phase=current_phase_type,
                    start_time=phase_start_time,
                    end_time=last_state.timestamp,
                    actions=actions,
                    initial_state=current_phase_states[0],
                    final_state=current_phase_states[-1],
                )
                timeline.append(phase_data)

                # Start New Phase
                current_phase_states = [state]
                current_phase_type = state.phase
                phase_start_time = state.timestamp
            else:
                current_phase_states.append(state)

            last_state = state

        # Close Final Phase
        if current_phase_states:
            actions = self._detect_actions_in_sequence(current_phase_states)
            phase_data = PhaseData(
                phase=current_phase_type,
                start_time=phase_start_time,
                end_time=last_state.timestamp,
                actions=actions,
                initial_state=current_phase_states[0],
                final_state=current_phase_states[-1],
            )
            timeline.append(phase_data)

        return timeline

    def _detect_actions_in_sequence(self, states: list[GameState]) -> list[ActionEvent]:
        """
        Detect actions by comparing consecutive states in a phase.
        """
        actions = []
        if len(states) < 2:
            return []

        prev = states[0]
        for curr in states[1:]:
            # Use ActionDetectionAgent logic (which is now pure logic, no LLM)
            detected = self.action_agent.detect_actions(prev, curr, curr.timestamp)

            for act in detected:
                logger.info(
                    f"Action Detected: {act.player_name} {act.action_type} {act.amount}"
                )
                # Micro-gestures are already in the BoardAgent output (ActionDetectionAgent logic propagates them)
                # No need to trigger FaceAgent separately.

                actions.append(act)

            prev = curr
        return actions

    def _get_frame_path(self, frames_dir, idx):
        # Assumes format frame_0000.jpg
        return os.path.join(frames_dir, f"frame_{idx:04d}.jpg")


if __name__ == "__main__":
    # Example Usage
    pipeline = PerceptionPipeline()
    # pipeline.run(...)
