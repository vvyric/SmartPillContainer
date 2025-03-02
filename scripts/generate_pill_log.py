import random
from datetime import datetime, timedelta

def generate_synthetic_pill_data(year=2025, month=1, output_file="synthetic_pill_log.log"):
    """
    Generate synthetic pill monitoring data for a month and save to a local file.
    
    Parameters:
    - year: The year for the data
    - month: The month for the data
    - output_file: The file path where the log will be saved
    """
    # Set random seed for reproducibility
    random.seed(2025)
    
    # Number of days in the month
    if month in [4, 6, 9, 11]:
        days_in_month = 30
    elif month == 2:
        # Check for leap year
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_month = 29
        else:
            days_in_month = 28
    else:
        days_in_month = 31
    
    log_data = []
    
    # For each day in the month
    for day in range(1, days_in_month + 1):
        # Randomly decide how many pills are missed (0, 1, or 2)
        # Most days should have all 3 pills taken
        missed_pills = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
        pills_to_take = 3 - missed_pills
        
        # Create time windows for morning, afternoon, and evening doses
        time_windows = [
            (datetime(year, month, day, 7, 0), datetime(year, month, day, 10, 0)),  # Morning: 7-10 AM
            (datetime(year, month, day, 12, 0), datetime(year, month, day, 15, 0)),  # Afternoon: 12-3 PM
            (datetime(year, month, day, 18, 0), datetime(year, month, day, 22, 0))   # Evening: 6-10 PM
        ]
        
        # Randomly select which doses to skip if any
        if missed_pills > 0:
            dose_indices = list(range(3))
            random.shuffle(dose_indices)
            dose_indices = dose_indices[:pills_to_take]
            dose_indices.sort()  # Keep them in chronological order
        else:
            dose_indices = [0, 1, 2]  # All doses
        
        # Generate pill events for the taken doses
        for dose_idx in dose_indices:
            start_time, end_time = time_windows[dose_idx]
            
            # Generate a random time within the window
            pill_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
            
            # Random values for the pill event
            tilt = random.uniform(15, 50)
            roll = random.uniform(-30, 40)
            pitch = random.uniform(-20, 10)
            yaw = random.uniform(40, 90)
            pressure = random.randint(490, 640)
            light = random.uniform(150, 670)
            
            # Time offset from start of event to details (in seconds)
            time_offset = random.uniform(20, 130)
            
            # Generate the initial pill event
            pill_event_time = pill_time
            pill_event_str = f"{pill_event_time.strftime('%Y-%m-%d %H:%M:%S')} - PILL EVENT DETECTED"
            log_data.append(pill_event_str)
            
            # Format matches the provided log
            log_data.append(f"Pill Event: Time {int(time_offset // 60)}m {time_offset % 60:.1f}s")
            
            # Pick-Place window
            pick_place_start = time_offset - random.uniform(1, 5)
            pick_place_end = time_offset + random.uniform(1, 5)
            log_data.append(f"  - Within Pick-Place window: {pick_place_start:.1f}s to {pick_place_end:.1f}s")
            
            # Add other sensor data
            log_data.append(f"  - 3D Tilt: {tilt:.2f}")
            log_data.append(f"  - Orientation: Roll={roll:.2f}°, Pitch={pitch:.2f}°, Yaw={yaw:.2f}°")
            log_data.append(f"  - Pressure: {pressure} at {pick_place_start + random.uniform(0, 3):.1f}s")
            log_data.append(f"  - Light: {light:.2f} lux at {time_offset:.1f}s")
            log_data.append(f"  - Awaiting swallow confirmation (press and hold pressure sensor within 10s)")
            log_data.append(f"{pill_event_time.strftime('%Y-%m-%d %H:%M:%S')} - Actual time of event: {pill_event_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Randomly decide if this is confirmed or forgotten (but taken)
            is_confirmed = random.choices([True, False], weights=[0.85, 0.15])[0]
            
            if is_confirmed:
                # Swallow confirmation after a short delay
                confirm_delay = random.uniform(2.5, 6.5)
                confirm_time = pill_time + timedelta(seconds=int(confirm_delay))
                log_data.append(f"{confirm_time.strftime('%Y-%m-%d %H:%M:%S')} - SWALLOW CONFIRMED")
                total_time = time_offset + confirm_delay
                log_data.append(f"PILL SWALLOW CONFIRMED: Time {int(total_time // 60)}m {total_time % 60:.1f}s")
                log_data.append(f"  - Pill taken at: {time_offset:.1f}s")
                log_data.append(f"  - Confirmation delay: {confirm_delay:.1f}s")
                log_data.append(f"{confirm_time.strftime('%Y-%m-%d %H:%M:%S')} - Actual time of swallow confirmation: {confirm_time.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # Forgotten pill (taken but not confirmed) after 10s timeout
                forget_time = pill_time + timedelta(seconds=10)
                log_data.append(f"{forget_time.strftime('%Y-%m-%d %H:%M:%S')} - POSSIBLE FORGOTTEN PILL")
                total_time = time_offset + 10.0
                log_data.append(f"POSSIBLE FORGOTTEN PILL: Time {int(total_time // 60)}m {total_time % 60:.1f}s")
                log_data.append(f"  - Pill taken at: {time_offset:.1f}s")
                log_data.append(f"  - No swallow confirmation received within 10.0s window")
                log_data.append(f"{forget_time.strftime('%Y-%m-%d %H:%M:%S')} - Actual time of forget event: {forget_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Now we need to handle missed doses - we'll create a record for tracking purposes
        if missed_pills > 0:
            # Identify which doses were missed
            all_indices = set([0, 1, 2])
            missed_indices = all_indices - set(dose_indices)
            
            # For the purpose of this log file, we don't need to include a marker for missed pills
            # as the actual log data only includes entries for pill events and confirmations/forgotten pills
    
    # Save the log data to a file
    with open(output_file, "w") as f:
        f.write("\n".join(log_data))
    
    print(f"Synthetic pill log data has been saved to: {output_file}")
    print(f"Generated {len(log_data)} lines of log data")
    
    # Calculate some basic statistics for reference
    total_pills = len([line for line in log_data if "PILL EVENT DETECTED" in line])
    confirmed_pills = len([line for line in log_data if "SWALLOW CONFIRMED" in line])
    forgotten_pills = len([line for line in log_data if "POSSIBLE FORGOTTEN PILL" in line])
    
    print(f"\nSummary Statistics:")
    print(f"Total Pill Events: {total_pills}")
    print(f"Confirmed Pills: {confirmed_pills}")
    print(f"Forgotten Pills: {forgotten_pills}")
    print(f"Missed Pills: 3 * {days_in_month} - {total_pills} = {3 * days_in_month - total_pills}")
    
    # Return the total number of entries generated
    return len(log_data)

# Run the generator with default settings (March 2025)
if __name__ == "__main__":
    # You can change the output file path here
    output_file = "synthetic_pill_log_jan2025.log"
    
    num_entries = generate_synthetic_pill_data(output_file=output_file)
    
    # Print a sample of the generated log (first 10 lines)
    print("\nSample of the generated log (first 10 lines):")
    with open(output_file, "r") as f:
        for i, line in enumerate(f):
            if i < 10:
                print(line.strip())
            else:
                break