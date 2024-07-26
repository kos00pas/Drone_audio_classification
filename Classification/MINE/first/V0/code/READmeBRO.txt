What is happen here ? 

1. the folders contain all the recording's (1 second length) and for each of them we have: 
	a. signal.csv
	b. mfcc.csv

2. Here the goal was to create for each of the recording the corresponding label.csv
	---STEPS:
	  --a. "r0_find_the_directories_for_saving.py"
		    1.  Find all the directories and save them in file: 
			     -> all_paths_an_labels.csv
			2. Give for all of them the label 'drone'
			3. Change one-by-one from 'drone' to 'not_drone' by hearing the recordings 		
	  --b. " r1_create_drone_not_Drone_csv.py"
			->create "drone.csv " and "not_drone.csv" 
			-> print the number of them :
					"""Number of occurrences of 'drone': 25371
						Number of occurrences of 'not_drone': 19151
								drone.csv and not_drone.csv created successfully."
	 --c. "r2_create_label_file_for_each.py"
			-> go to every subdirectory and create the "label.csv" 
							->"label.csv" contains the string 'drone' and 'not_drone' respectively
	 
	