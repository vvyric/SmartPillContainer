# CACOM
Repo for project of CACOM: A Smart Pill Container for Accurate Pill Retrieval Detection and Disease Control

## Folder Description:
```hardware```:  folder contains codes for hardware.

```notebook```: basic analysis of raw data of hardware.

```plots```: plots of data analysis.

```log``` the log the alghorithm save.

```data``` save the recorded raw data from the sensor.

```scripts```: 
    - ```event_detenction.py```: normal event detection without swallow confirmation
    - ```event_detection_swallow_confirmation.py```: enhanced event detection with swallow confimation
    - ```generate_pill_log.py```: generate synthetic data for statistical analysis
    - ```pill_taken_analyze.py```: analyze the log.
    
```ros_ws```: ros package of the algorithm, detailed description see the ```README.md``` in ```rosws/src/pill_detection```.
