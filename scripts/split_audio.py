import os
from pathlib import Path
from pydub import AudioSegment
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_timestamp(ts_str):
    """Convert timestamp string to milliseconds"""
    try:
        dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
        return (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
    except ValueError as e:
        logger.error(f"Error parsing timestamp {ts_str}: {e}")
        return None


def format_timestamp(ms):
    """Convert milliseconds to HH:MM:SS.mmm format"""
    hours = ms // (3600 * 1000)
    ms = ms % (3600 * 1000)
    minutes = ms // (60 * 1000)
    ms = ms % (60 * 1000)
    seconds = ms // 1000
    ms = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def get_output_filename(base_name, phase):
    """Generate output filename based on phase"""
    # Remove the last suffix for all phases
    parts = base_name.split("-")
    base_name = "-".join(parts[:-2])
    match parts[-2]:
        case "C":  # congruent => related (based on readmes)
            base_name = f"{base_name}-R"
        case "I":  # incongruent => unrelated (based on readmes)
            base_name = f"{base_name}-U"
        case "M":  # missing => not present (based on readmes)
            base_name = f"{base_name}-N"

    return f"{base_name}.wav"


def is_file_processed(audio_dir, base_name, phase):
    """Check if a file has already been processed"""
    output_filename = get_output_filename(base_name, phase)
    output_path = os.path.join(audio_dir, phase, output_filename)
    return os.path.exists(output_path)


def setup_audio_directories():
    """Setup the audio directory structure and remove old directories"""
    audio_dir = "../ufal_emmt/preprocessed-data/audio"
    phases = ["Read", "Translate", "See", "Update"]

    # Create main audio directory
    os.makedirs(audio_dir, exist_ok=True)

    # Create phase directories
    for phase in phases:
        os.makedirs(os.path.join(audio_dir, phase), exist_ok=True)

    return audio_dir


def process_probe_dir(probe_dir, audio_dir):
    try:
        # Find all .events files
        events_files = list(Path(probe_dir).glob("*.events"))
        logger.info(f"Found {len(events_files)} events files in {probe_dir}")

        for events_file in events_files:
            try:
                # Get the base name without extension
                base_name = events_file.stem
                logger.info(f"Processing {base_name}")

                # Check if all phases are already processed
                if all(
                    is_file_processed(audio_dir, base_name, phase) for phase in ["Read", "Translate", "See", "Update"]
                ):
                    logger.info(f"Skipping {base_name} - already processed")
                    continue

                # Find corresponding MP3 file
                mp3_file = events_file.with_suffix(".mp3")
                if not mp3_file.exists():
                    logger.warning(f"No MP3 file found for {events_file}")
                    continue

                # Read the events file
                try:
                    with open(events_file, "r") as f:
                        content = f.read()
                except IOError as e:
                    logger.error(f"Error reading events file {events_file}: {e}")
                    continue

                # Find all relevant timestamps
                timestamps = {}
                for line in content.split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        event = parts[0].strip()
                        ts_str = parts[1].strip()

                        if event.endswith(". start"):  # event == "target text presented":
                            timestamps["START_READ"] = parse_timestamp(ts_str)
                            logger.info(f"Found START_READ at {ts_str}")
                        elif event == "start response recording":
                            timestamps["END_READ"] = parse_timestamp(ts_str)
                            timestamps["START_TRANSLATE"] = parse_timestamp(ts_str)
                            logger.info(f"Found END_READ/START_TRANSLATE at {ts_str}")
                        elif event == "end response recording":
                            if "END_TRANSLATE" not in timestamps:
                                timestamps["END_TRANSLATE"] = parse_timestamp(ts_str)
                                logger.info(f"Found END_TRANSLATE at {ts_str}")
                            else:
                                timestamps["END_UPDATE"] = parse_timestamp(ts_str)
                                logger.info(f"Found END_UPDATE at {ts_str}")
                        elif event == "present image":
                            timestamps["START_SEE"] = parse_timestamp(ts_str)
                            logger.info(f"Found START_SEE at {ts_str}")
                        elif event == "record translation with image":
                            timestamps["END_SEE"] = parse_timestamp(ts_str)
                            timestamps["START_UPDATE"] = parse_timestamp(ts_str)
                            logger.info(f"Found END_SEE/START_UPDATE at {ts_str}")
                        elif event.endswith(". end"):
                            timestamps["END_UPDATE"] = parse_timestamp(ts_str)
                            logger.info(f"Found END_UPDATE at {ts_str}")

                if not timestamps:
                    logger.warning(f"No timestamps found in {events_file}")
                    continue

                # Normalize all timestamps by subtracting START_READ
                if "START_READ" in timestamps:
                    start_read = timestamps["START_READ"]
                    logger.info(f"Normalizing timestamps by subtracting START_READ: {format_timestamp(start_read)}")
                    for key in timestamps:
                        timestamps[key] -= start_read
                        logger.info(f"Normalized {key}: {format_timestamp(timestamps[key])}")
                else:
                    logger.error(f"No START_READ timestamp found in {events_file}")
                    continue

                # Load the MP3 file
                try:
                    audio = AudioSegment.from_mp3(mp3_file)
                    logger.info(f"Loaded MP3 file: {mp3_file}, duration: {len(audio)}ms")
                except Exception as e:
                    logger.error(f"Error loading MP3 file {mp3_file}: {e}")
                    continue

                # Split the audio based on timestamps and convert to WAV
                try:
                    # READ segment
                    if (
                        "START_READ" in timestamps
                        and "END_READ" in timestamps
                        and not is_file_processed(audio_dir, base_name, "read")
                    ):
                        start_ms = timestamps["START_READ"]
                        end_ms = timestamps["END_READ"]
                        if start_ms >= end_ms:
                            logger.error(f"Invalid READ segment: start ({start_ms}ms) >= end ({end_ms}ms)")
                            continue
                        read_segment = audio[start_ms:end_ms]
                        logger.info(f"Created READ segment: {len(read_segment)}ms ({start_ms}ms to {end_ms}ms)")
                        output_filename = get_output_filename(base_name, "read")
                        output_path = os.path.join(audio_dir, "read", output_filename)
                        read_segment.export(output_path, format="wav")
                        logger.info(f"Created {output_path}")

                    # TRANSLATE segment
                    if (
                        "START_TRANSLATE" in timestamps
                        and "END_TRANSLATE" in timestamps
                        and not is_file_processed(audio_dir, base_name, "translate")
                    ):
                        start_ms = timestamps["START_TRANSLATE"]
                        end_ms = timestamps["END_TRANSLATE"]
                        if start_ms >= end_ms:
                            logger.error(f"Invalid TRANSLATE segment: start ({start_ms}ms) >= end ({end_ms}ms)")
                            continue
                        translate_segment = audio[start_ms:end_ms]
                        logger.info(
                            f"Created TRANSLATE segment: {len(translate_segment)}ms ({start_ms}ms to {end_ms}ms)"
                        )
                        output_filename = get_output_filename(base_name, "translate")
                        output_path = os.path.join(audio_dir, "translate", output_filename)
                        translate_segment.export(output_path, format="wav")
                        logger.info(f"Created {output_path}")

                    # SEE segment
                    if (
                        "START_SEE" in timestamps
                        and "END_SEE" in timestamps
                        and not is_file_processed(audio_dir, base_name, "see")
                    ):
                        start_ms = timestamps["START_SEE"]
                        end_ms = timestamps["END_SEE"]
                        if start_ms >= end_ms:
                            logger.error(f"Invalid SEE segment: start ({start_ms}ms) >= end ({end_ms}ms)")
                            continue
                        see_segment = audio[start_ms:end_ms]
                        logger.info(f"Created SEE segment: {len(see_segment)}ms ({start_ms}ms to {end_ms}ms)")
                        output_filename = get_output_filename(base_name, "see")
                        output_path = os.path.join(audio_dir, "see", output_filename)
                        see_segment.export(output_path, format="wav")
                        logger.info(f"Created {output_path}")

                    # UPDATE segment
                    if (
                        "START_UPDATE" in timestamps
                        and "END_UPDATE" in timestamps
                        and not is_file_processed(audio_dir, base_name, "update")
                    ):
                        start_ms = timestamps["START_UPDATE"]
                        end_ms = timestamps["END_UPDATE"]
                        if start_ms >= end_ms:
                            logger.error(f"Invalid UPDATE segment: start ({start_ms}ms) >= end ({end_ms}ms)")
                            continue
                        update_segment = audio[start_ms:end_ms]
                        logger.info(f"Created UPDATE segment: {len(update_segment)}ms ({start_ms}ms to {end_ms}ms)")
                        output_filename = get_output_filename(base_name, "update")
                        output_path = os.path.join(audio_dir, "update", output_filename)
                        update_segment.export(output_path, format="wav")
                        logger.info(f"Created {output_path}")
                except Exception as e:
                    logger.error(f"Error processing audio segments for {base_name}: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error processing file {events_file}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error processing directory {probe_dir}: {e}")


def main():
    base_dir = "../ufal_emmt/probes"

    try:
        # Setup audio directories
        audio_dir = setup_audio_directories()

        # Process each probe directory
        for probe_dir in Path(base_dir).glob("probe*"):
            if probe_dir.is_dir():
                logger.info(f"Processing {probe_dir}")
                process_probe_dir(str(probe_dir), audio_dir)
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
