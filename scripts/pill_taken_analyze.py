import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns
import os

def read_pill_log(log_file_path, expected_daily_pills=3):
    """
    Read and parse a pill monitoring log file into structured data.
    
    Parameters:
    - log_file_path: Path to the log file
    - expected_daily_pills: Number of pills expected to be taken daily (default: 3)
    
    Returns:
    - log_data: List of original log lines
    - df: DataFrame with structured data for analysis
    """
    print(f"Reading log file: {log_file_path}")
    
    # Check if file exists
    if not os.path.exists(log_file_path):
        print(f"Error: File '{log_file_path}' does not exist.")
        return [], pd.DataFrame()
    
    # Read the log file
    with open(log_file_path, 'r') as f:
        log_data = f.read().splitlines()
    
    print(f"Successfully read {len(log_data)} lines from the log file")
    
    # Initialize lists to store the extracted data
    event_data = []
    
    # Initialize tracking variables
    current_event_time = None
    current_event_details = {}
    event_index = 0
    
    # Process each line in the log
    while event_index < len(log_data):
        # Look for a pill event
        if " - PILL EVENT DETECTED" in log_data[event_index]:
            # Extract the timestamp
            timestamp_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_data[event_index])
            if timestamp_match:
                # Start a new event
                current_event_time = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                
                # Initialize event details
                current_event_details = {
                    'date': current_event_time.date(),
                    'datetime': current_event_time,
                    'time_of_day': current_event_time.hour + current_event_time.minute / 60.0,
                    'status': None,  # Will be set later when we find confirmation
                    'tilt': None,
                    'roll': None,
                    'pitch': None,
                    'yaw': None,
                    'pressure': None,
                    'light': None,
                    'confirmation_delay': None
                }
                
                # Determine dose time based on hour
                hour = current_event_time.hour
                if 5 <= hour < 11:
                    current_event_details['dose_time'] = "Morning"
                elif 11 <= hour < 17:
                    current_event_details['dose_time'] = "Afternoon"
                else:
                    current_event_details['dose_time'] = "Evening"
                
                # Move to next line to begin reading event details
                event_index += 1
                
                # Process event details until we find a SWALLOW CONFIRMED or POSSIBLE FORGOTTEN PILL
                while event_index < len(log_data):
                    line = log_data[event_index]
                    
                    # Check if we've found a confirmation or forgotten pill
                    if " - SWALLOW CONFIRMED" in line:
                        # Extract confirmation timestamp
                        confirm_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                        if confirm_match:
                            confirm_time = datetime.strptime(confirm_match.group(1), "%Y-%m-%d %H:%M:%S")
                            current_event_details['status'] = 'swallow'
                            # Look for confirmation delay in next lines
                            next_line_index = event_index + 1
                            if next_line_index < len(log_data) and "Confirmation delay:" in log_data[next_line_index + 1]:
                                delay_match = re.search(r"Confirmation delay: ([\d.]+)s", log_data[next_line_index + 1])
                                if delay_match:
                                    current_event_details['confirmation_delay'] = float(delay_match.group(1))
                            else:
                                # If not found, calculate from timestamps
                                current_event_details['confirmation_delay'] = (confirm_time - current_event_time).total_seconds()
                            
                            # Add this event to our dataset
                            event_data.append(current_event_details)
                            break
                    
                    elif " - POSSIBLE FORGOTTEN PILL" in line:
                        current_event_details['status'] = 'forgotten'
                        # Add this event to our dataset
                        event_data.append(current_event_details)
                        break
                    
                    # Extract event details
                    elif "3D Tilt:" in line:
                        tilt_match = re.search(r"3D Tilt: ([\d.]+)", line)
                        if tilt_match:
                            current_event_details['tilt'] = float(tilt_match.group(1))
                    
                    elif "Orientation:" in line:
                        orientation_match = re.search(r"Roll=([-\d.]+)°, Pitch=([-\d.]+)°, Yaw=([-\d.]+)°", line)
                        if orientation_match:
                            current_event_details['roll'] = float(orientation_match.group(1))
                            current_event_details['pitch'] = float(orientation_match.group(2))
                            current_event_details['yaw'] = float(orientation_match.group(3))
                    
                    elif "Pressure:" in line:
                        pressure_match = re.search(r"Pressure: (\d+)", line)
                        if pressure_match:
                            current_event_details['pressure'] = int(pressure_match.group(1))
                    
                    elif "Light:" in line:
                        light_match = re.search(r"Light: ([\d.]+)", line)
                        if light_match:
                            current_event_details['light'] = float(light_match.group(1))
                    
                    event_index += 1
            
        event_index += 1
    
    # Convert to DataFrame
    print(f"Extracted {len(event_data)} pill events from the log")
    
    if not event_data:
        return log_data, pd.DataFrame()
    
    df = pd.DataFrame(event_data)
    
    # Sort by datetime
    df = df.sort_values('datetime')
    
    # Now we need to infer missed pills
    print("Inferring missed pills based on daily pill count...")
    
    # For each day in the dataset, count how many pills were taken and add "missed" entries
    days = df['date'].unique()
    missed_data = []
    
    for day in days:
        day_df = df[df['date'] == day]
        actual_count = len(day_df)
        
        # If fewer pills were taken than expected, add missed pill entries
        if actual_count < expected_daily_pills:
            missing_count = expected_daily_pills - actual_count
            print(f"Day {day}: Found {actual_count} pills, inferring {missing_count} missed pills")
            
            # Determine which dose times are already covered
            covered_dose_times = set(day_df['dose_time'])
            possible_dose_times = {"Morning", "Afternoon", "Evening"}
            missing_dose_times = possible_dose_times - covered_dose_times
            
            # If we still need more missed entries after accounting for missing dose times,
            # just make them up based on typical timing
            if len(missing_dose_times) < missing_count:
                # This is a fallback if our dose time classification isn't perfect
                print(f"Warning: Could not perfectly classify missing doses for {day}. Using best guess.")
                remaining_needed = missing_count - len(missing_dose_times)
                
                # Find the dose times that have the fewest entries
                dose_time_counts = day_df['dose_time'].value_counts().to_dict()
                for dose_time in possible_dose_times:
                    if dose_time not in dose_time_counts:
                        dose_time_counts[dose_time] = 0
                
                sorted_dose_times = sorted(possible_dose_times, key=lambda dt: dose_time_counts[dt])
                for dose_time in sorted_dose_times:
                    if dose_time not in missing_dose_times and remaining_needed > 0:
                        missing_dose_times.add(dose_time)
                        remaining_needed -= 1
            
            # Add the missed pill entries
            for i, dose_time in enumerate(list(missing_dose_times)[:missing_count]):
                # Generate a reasonable time for the missed dose
                if dose_time == "Morning":
                    hour = 8  # 8:00 AM
                elif dose_time == "Afternoon":
                    hour = 13  # 1:00 PM
                else:  # Evening
                    hour = 19  # 7:00 PM
                
                # Create a timestamp for the missed pill
                missed_time = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                
                # Add a missed pill entry
                missed_data.append({
                    'date': day,
                    'datetime': missed_time,
                    'time_of_day': hour,
                    'dose_time': dose_time,
                    'status': 'missed',
                    'tilt': None,
                    'roll': None,
                    'pitch': None,
                    'yaw': None,
                    'pressure': None,
                    'light': None,
                    'confirmation_delay': None
                })
                
                print(f"- Inferred missed {dose_time} pill on {day}")
    
    # If we have any missed pills, add them to the DataFrame
    if missed_data:
        missed_df = pd.DataFrame(missed_data)
        df = pd.concat([df, missed_df], ignore_index=True)
        
        # Resort by datetime after adding missed pills
        df = df.sort_values('datetime')
    
    print(f"Final dataset contains {len(df)} events")
    return log_data, df

def analyze_pill_data(df, expected_daily_pills=3):
    """
    Analyze the pill data and calculate summary statistics.
    """
    if df.empty:
        print("No data to analyze")
        return None, {
            'num_days': 0,
            'perfect_days': 0,
            'days_with_1_missed': 0,
            'days_with_2_missed': 0,
            'days_with_all_missed': 0,
            'total_swallow': 0,
            'total_forgotten': 0,
            'total_missed': 0,
            'expected_total': 0,
            'overall_adherence': 0,
            'confirmed_rate': 0
        }
    
    # Group by date and status to get daily counts
    daily_status = df.groupby(['date', 'status']).size().unstack(fill_value=0)
    
    # Ensure all statuses are present
    for status in ['swallow', 'forgotten', 'missed']:
        if status not in daily_status.columns:
            daily_status[status] = 0
    
    # Calculate total pills taken (swallow + forgotten)
    daily_status['total_taken'] = daily_status['swallow'] + daily_status['forgotten']
    daily_status['expected'] = expected_daily_pills  # Expected pills per day
    daily_status['adherence_rate'] = (daily_status['total_taken'] / daily_status['expected']) * 100
    
    # Summary statistics
    num_days = len(daily_status)
    total_pills_taken = daily_status['total_taken'].sum()
    total_swallow = daily_status['swallow'].sum()
    total_forgotten = daily_status['forgotten'].sum()
    total_missed = daily_status['missed'].sum()
    expected_total = num_days * expected_daily_pills
    
    # Calculate days with different adherence levels
    perfect_days = (daily_status['total_taken'] == expected_daily_pills).sum()
    days_with_1_missed = (daily_status['total_taken'] == expected_daily_pills - 1).sum()
    days_with_2_missed = (daily_status['total_taken'] == expected_daily_pills - 2).sum()
    days_with_all_missed = (daily_status['total_taken'] == 0).sum()
    
    overall_adherence = (total_pills_taken / expected_total) * 100
    confirmed_rate = (total_swallow / expected_total) * 100
    
    summary = {
        'num_days': num_days,
        'perfect_days': perfect_days,
        'days_with_1_missed': days_with_1_missed,
        'days_with_2_missed': days_with_2_missed,
        'days_with_all_missed': days_with_all_missed,
        'total_swallow': total_swallow,
        'total_forgotten': total_forgotten,
        'total_missed': total_missed,
        'expected_total': expected_total,
        'overall_adherence': overall_adherence,
        'confirmed_rate': confirmed_rate
    }
    
    return daily_status, summary

def visualize_pill_data(df, daily_status, summary):
    """
    Create comprehensive visualizations of pill data.
    """
    if df.empty or daily_status is None:
        print("No data to visualize")
        return None
    
    # Determine the month and year from the data
    start_date = df['datetime'].min()
    year = start_date.year
    month = start_date.month
    
    # Set up the figure with subplots
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 12))
    spec = gridspec.GridSpec(ncols=1, nrows=3, height_ratios=[4, 3, 3])
    
    # 1. Scatter plot of pill events by time of day
    ax1 = fig.add_subplot(spec[0])
    
    # Plot points with different markers/colors for each status
    if 'swallow' in df['status'].unique():
        ax1.scatter(df[df['status'] == 'swallow']['datetime'], 
                    df[df['status'] == 'swallow']['time_of_day'],
                    color='green', label='Swallow Confirmed', marker='o', s=80, alpha=0.7)
    
    if 'forgotten' in df['status'].unique():
        ax1.scatter(df[df['status'] == 'forgotten']['datetime'], 
                    df[df['status'] == 'forgotten']['time_of_day'],
                    color='orange', label='Possible Forgotten Pill', marker='s', s=80, alpha=0.7)
    
    if 'missed' in df['status'].unique():
        ax1.scatter(df[df['status'] == 'missed']['datetime'], 
                    df[df['status'] == 'missed']['time_of_day'],
                    color='red', label='Missed Pill', marker='x', s=100, alpha=0.7)
    
    # Format x-axis to show dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    # Format y-axis for hours
    hour_ticks = range(6, 24, 3)
    ax1.set_yticks(hour_ticks)
    ax1.set_yticklabels([f"{h:02d}:00" for h in hour_ticks])
    ax1.set_ylim(5, 23)  # Focus on waking hours
    
    # Add grid and labels
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Time of Day', fontsize=12)
    ax1.set_title(f'Pill Events by Time of Day - {datetime(year, month, 1).strftime("%B %Y")}', fontsize=14)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 2. Daily pill counts - stacked bar chart
    ax2 = fig.add_subplot(spec[1])
    
    # Prepare data for stacked bar
    daily_data = daily_status.reset_index()
    
    # Create the stacked bar chart
    ax2.bar(daily_data['date'], daily_data['swallow'], label='Confirmed', color='green')
    ax2.bar(daily_data['date'], daily_data['forgotten'], bottom=daily_data['swallow'], 
            label='Forgotten', color='orange')
    
    # Add a line for expected pills
    expected_pills = daily_data['expected'].iloc[0]  # Assuming same expectation for all days
    ax2.axhline(y=expected_pills, color='red', linestyle='--', label=f'Expected ({expected_pills} pills)')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    
    # Add grid and labels
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Number of Pills', fontsize=12)
    ax2.set_title('Daily Pill Intake', fontsize=14)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 3. Summary statistics table
    ax3 = fig.add_subplot(spec[2])
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create summary table data
    summary_data = [
        ["Total Days", summary['num_days']],
        ["Perfect Days (All Pills)", summary['perfect_days']],
        ["Days with 1 Missed Pill", summary['days_with_1_missed']],
        ["Days with 2 Missed Pills", summary['days_with_2_missed']],
        ["Days with All Pills Missed", summary['days_with_all_missed']],
        ["Total Confirmed Pills", summary['total_swallow']],
        ["Total Forgotten Pills", summary['total_forgotten']],
        ["Total Missed Pills", summary['total_missed']],
        ["Expected Total Pills", summary['expected_total']],
        ["Overall Adherence Rate", f"{summary['overall_adherence']:.1f}%"],
        ["Confirmed Pills Rate", f"{summary['confirmed_rate']:.1f}%"]
    ]
    
    # Set up the table
    table = ax3.table(cellText=summary_data, 
                      colLabels=["Metric", "Value"],
                      cellLoc='center', 
                      loc='center',
                      colWidths=[0.6, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Style the header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(2)):
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontweight='bold')
    
    ax3.set_title("Medication Adherence Summary", fontsize=14, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    return fig

def create_heatmap_calendar(daily_status):
    """
    Create a calendar heatmap showing pill adherence by day.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    from datetime import datetime, date
    
    if daily_status is None or daily_status.empty:
        print("No data to create calendar heatmap")
        return None
    
    # Process the data
    month_data = daily_status.copy()
    
    # Convert index to proper datetime if it's not already
    if not isinstance(month_data.index, pd.DatetimeIndex):
        print(f"Index type: {type(month_data.index)}")
        print(f"First index item: {month_data.index[0]}, type: {type(month_data.index[0])}")
        
        # Try to convert the index to datetime
        try:
            month_data.index = pd.to_datetime(month_data.index)
            print("Successfully converted index to datetime")
        except Exception as e:
            print(f"Error converting index to datetime: {e}")
            print("Calendar heatmap cannot be created.")
            return None
    
    # Now extract day and weekday
    try:
        # Extract day of month and weekday manually
        days = []
        weekdays = []
        for idx in month_data.index:
            days.append(idx.day)
            weekdays.append(idx.weekday())
        
        month_data['day'] = days
        month_data['weekday'] = weekdays
        
        # Determine year and month from the data
        first_date = month_data.index[0]
        year = first_date.year
        month = first_date.month
        
        print(f"Extracted days for {month}/{year} calendar")
    except Exception as e:
        print(f"Error extracting day information: {e}")
        print("Calendar heatmap cannot be created.")
        return None
    
    # Calculate a custom position for calendar layout
    # First, find first day of month weekday
    first_day = datetime(year, month, 1)
    first_weekday = first_day.weekday()  # 0 = Monday, 6 = Sunday
    
    # Create a day-to-position mapping
    day_positions = {}
    day = 1
    for week in range(6):  # Max 6 weeks in a month
        for weekday in range(7):  # 7 days in a week
            if week == 0 and weekday < first_weekday:
                continue
            if day > max(days):
                break
            day_positions[day] = (week, weekday)
            day += 1
    
    # Create a matrix for the heatmap
    if not day_positions:
        print("Error: No day positions could be calculated")
        return None
        
    num_weeks = max(week for week, _ in day_positions.values()) + 1
    calendar_data = np.zeros((num_weeks, 7))
    calendar_data.fill(np.nan)  # Fill with NaN for days not in the month
    
    # Fill the matrix with adherence data
    for day, adherence in zip(month_data['day'], month_data['adherence_rate']):
        if day in day_positions:
            week, weekday = day_positions[day]
            calendar_data[week, weekday] = adherence
    
    # Create the heatmap figure
    plt.figure(figsize=(12, 8))
    
    # Custom colormap: red (0%) to yellow (50%) to green (100%)
    cmap = ListedColormap(['#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00', 
                           '#bfff00', '#80ff00', '#40ff00', '#00ff00'])
    
    # Plot the heatmap
    ax = sns.heatmap(calendar_data, cmap=cmap, vmin=0, vmax=100, 
                    linewidths=1, linecolor='black',
                    cbar_kws={'label': 'Adherence Rate (%)'})
    
    # Set the labels
    weekdays_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticklabels(weekdays_labels)
    ax.set_yticklabels([f'Week {i+1}' for i in range(num_weeks)], rotation=0)
    
    # Add day numbers to each cell
    for day, (week, weekday) in day_positions.items():
        ax.text(weekday + 0.5, week + 0.5, str(day), 
                ha='center', va='center', fontsize=11,
                color='black' if calendar_data[week, weekday] > 50 else 'white')
    
    ax.set_title(f'Medication Adherence Calendar - {datetime(year, month, 1).strftime("%B %Y")}', fontsize=14)
    
    return plt.gcf()

# Alternative simplified function that doesn't rely on date/time operations
def create_simple_calendar_heatmap(daily_status):
    """
    Create a simplified calendar heatmap showing pill adherence by day.
    This version doesn't rely on datetime operations.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import calendar
    import seaborn as sns
    from matplotlib.colors import ListedColormap
    
    if daily_status is None or daily_status.empty:
        print("No data to create calendar heatmap")
        return None
    
    # Reset index to make the date a column
    month_data = daily_status.reset_index()
    
    # Print column names and first row for debugging
    print("Columns:", month_data.columns.tolist())
    print("First row:", month_data.iloc[0].tolist())
    
    # Assuming the date column is now named 'index' or 'date'
    date_col = 'index' if 'index' in month_data.columns else 'date'
    
    # Extract month and year from the first date
    # This assumes the date is in a format that can be split
    try:
        # Try different approaches to extract date information
        first_date_str = str(month_data[date_col].iloc[0])
        date_parts = first_date_str.split('-')
        
        if len(date_parts) >= 2:
            year = int(date_parts[0])
            month = int(date_parts[1])
        else:
            # Fallback to using the current date
            import datetime
            now = datetime.datetime.now()
            year = now.year
            month = now.month
        
        # Extract days
        month_data['day'] = month_data[date_col].astype(str).str.split('-').str[2].astype(int)
        
        print(f"Using year={year}, month={month}")
    except Exception as e:
        print(f"Error extracting date information: {e}")
        return None
    
    # Get the number of days in the month
    days_in_month = calendar.monthrange(year, month)[1]
    
    # Get the weekday of the first day (0 = Monday, 6 = Sunday)
    first_weekday = calendar.monthrange(year, month)[0]
    
    # Create a day-to-position mapping
    day_positions = {}
    day = 1
    for week in range(6):  # Max 6 weeks in a month
        for weekday in range(7):  # 7 days in a week
            if week == 0 and weekday < first_weekday:
                continue
            if day > days_in_month:
                break
            day_positions[day] = (week, weekday)
            day += 1
    
    # Create a matrix for the heatmap
    num_weeks = max(week for week, _ in day_positions.values()) + 1
    calendar_data = np.zeros((num_weeks, 7))
    calendar_data.fill(np.nan)  # Fill with NaN for days not in the month
    
    # Fill the matrix with adherence data
    for _, row in month_data.iterrows():
        day = row['day']
        adherence = row['adherence_rate']
        if day in day_positions:
            week, weekday = day_positions[day]
            calendar_data[week, weekday] = adherence
    
    # Create the heatmap figure
    plt.figure(figsize=(12, 8))
    
    # Custom colormap: red (0%) to yellow (50%) to green (100%)
    cmap = ListedColormap(['#ff0000', '#ff4000', '#ff8000', '#ffbf00', '#ffff00', 
                          '#bfff00', '#80ff00', '#40ff00', '#00ff00'])
    
    # Plot the heatmap
    ax = sns.heatmap(calendar_data, cmap=cmap, vmin=0, vmax=100, 
                    linewidths=1, linecolor='black',
                    cbar_kws={'label': 'Adherence Rate (%)'})
    
    # Set the labels
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticklabels(weekdays)
    ax.set_yticklabels([f'Week {i+1}' for i in range(num_weeks)], rotation=0)
    
    # Add day numbers to each cell
    for day, (week, weekday) in day_positions.items():
        ax.text(weekday + 0.5, week + 0.5, str(day), 
                ha='center', va='center', fontsize=11,
                color='black' if calendar_data[week, weekday] > 50 else 'white')
    
    # Get month name
    month_name = calendar.month_name[month]
    ax.set_title(f'Medication Adherence Calendar - {month_name} {year}', fontsize=14)
    
    return plt.gcf()

def analyze_dose_time_patterns(df):
    """
    Analyze patterns in dose timing and missed doses.
    """
    if df.empty:
        print("No data to analyze dose time patterns")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. Dose time distribution by status
    dose_status_counts = df.groupby(['dose_time', 'status']).size().unstack(fill_value=0)
    
    # Sort by typical order: Morning, Afternoon, Evening
    dose_order = ["Morning", "Afternoon", "Evening"]
    if set(dose_order).issubset(set(dose_status_counts.index)):
        dose_status_counts = dose_status_counts.reindex(dose_order)
    
    dose_status_counts.plot(kind='bar', stacked=True, ax=axes[0], 
                           color={'swallow': 'green', 'forgotten': 'orange', 'missed': 'red'})
    axes[0].set_title('Pill Status by Dose Time', fontsize=14)
    axes[0].set_xlabel('Dose Time')
    axes[0].set_ylabel('Number of Pills')
    axes[0].legend(['Confirmed', 'Forgotten', 'Missed'])
    axes[0].grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Calculate and display percentages above each bar
    for i, dose in enumerate(dose_status_counts.index):
        total = dose_status_counts.loc[dose].sum()
        for j, status in enumerate(['swallow', 'forgotten', 'missed']):
            if status in dose_status_counts.columns:
                count = dose_status_counts.loc[dose, status]
                if count > 0:
                    percentage = count / total * 100
                    # Place text at the middle of each segment
                    if j == 0:  # swallow
                        y_pos = count / 2
                    elif j == 1:  # forgotten
                        y_pos = dose_status_counts.loc[dose, 'swallow'] + count / 2
                    else:  # missed
                        y_pos = dose_status_counts.loc[dose, 'swallow'] + dose_status_counts.loc[dose, 'forgotten'] + count / 2
                    
                    axes[0].text(i, y_pos, f"{percentage:.1f}%", ha='center', va='center',
                                color='white' if j != 1 else 'black', fontweight='bold')
    
    # 2. Hour of day distribution by status
    # Create hour bins
    df['hour_bin'] = df['time_of_day'].apply(lambda x: int(x))
    hour_status_counts = df.groupby(['hour_bin', 'status']).size().unstack(fill_value=0)
    
    # Limit to waking hours (6 AM to 11 PM)
    waking_hours = range(6, 24)
    hour_status_counts = hour_status_counts.reindex(waking_hours, fill_value=0)
    
    # Plot hour distribution
    hour_status_counts.plot(kind='bar', stacked=True, ax=axes[1],
                           color={'swallow': 'green', 'forgotten': 'orange', 'missed': 'red'})
    axes[1].set_title('Pill Status by Hour of Day', fontsize=14)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Number of Pills')
    axes[1].legend(['Confirmed', 'Forgotten', 'Missed'])
    axes[1].grid(True, linestyle='--', alpha=0.7, axis='y')
    axes[1].set_xticklabels([f"{h:02d}:00" for h in waking_hours])
    
    plt.tight_layout()
    return fig

def main():
    # Ask for the log file path
    # log_file_path = input("Enter the path to the pill log file: ")
    log_file_path = "/workspaces/LongSequence_MasterThesis/CACOM/synthetic_pill_log_jan2025.log"
    
    # Ask for the expected number of daily pills
    try:
        expected_daily_pills = int(input("Enter the expected number of pills per day [3]: ") or "3")
    except ValueError:
        print("Invalid input. Using default value of 3 pills per day.")
        expected_daily_pills = 3
    
    # Read and parse the log file
    print(f"Reading and parsing the log file with expected {expected_daily_pills} pills per day...")
    log_data, df = read_pill_log(log_file_path, expected_daily_pills)
    
    # Analyze the data
    print("Analyzing pill data...")
    daily_status, summary = analyze_pill_data(df, expected_daily_pills)
    
    # Print summary statistics
    print("\nMedication Adherence Summary:")
    print(f"Total Days: {summary['num_days']}")
    print(f"Perfect Days (All Pills): {summary['perfect_days']}")
    print(f"Days with 1 Missed Pill: {summary['days_with_1_missed']}")
    print(f"Days with 2 Missed Pills: {summary['days_with_2_missed']}")
    print(f"Days with All Pills Missed: {summary['days_with_all_missed']}")
    print(f"Total Confirmed Pills: {summary['total_swallow']}")
    print(f"Total Forgotten Pills: {summary['total_forgotten']}")
    print(f"Total Missed Pills: {summary['total_missed']}")
    print(f"Expected Total Pills: {summary['expected_total']}")
    print(f"Overall Adherence Rate: {summary['overall_adherence']:.1f}%")
    print(f"Confirmed Pills Rate: {summary['confirmed_rate']:.1f}%")
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig = visualize_pill_data(df, daily_status, summary)
    
    if fig:
        # Save the visualization
        output_path = "pill_monitoring_summary.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Main visualization saved to: {output_path}")
    
    # Create calendar heatmap
    calendar_fig = create_heatmap_calendar(daily_status)
    
    if calendar_fig:
        # Save the calendar visualization
        calendar_path = "pill_monitoring_calendar.png"
        calendar_fig.savefig(calendar_path, dpi=300, bbox_inches='tight')
        print(f"Calendar visualization saved to: {calendar_path}")
    
    # Create dose time analysis
    dose_time_fig = analyze_dose_time_patterns(df)
    
    if dose_time_fig:
        # Save the dose time analysis
        dose_time_path = "pill_dose_time_analysis.png"
        dose_time_fig.savefig(dose_time_path, dpi=300, bbox_inches='tight')
        print(f"Dose time analysis saved to: {dose_time_path}")
    
    print("\nAnalysis complete. Visualizations saved.")
    
    # Show the visualizations
    plt.show()

if __name__ == "__main__":
    main()