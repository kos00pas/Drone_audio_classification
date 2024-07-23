order :

* load data in the folder data
* if tyour data is .mp3 then run the conversion_for_mp3.py

* resampling.py :   convert them to sample rate -> 16kHz
* splitting.py :Split them in 1sec duration
    - create_singal_csv.py : create a directory for each 1 sec  to contain :
        1. signal.csv (16000,1) ,dtype=np.int16 based on recording of respeaker dtype:float32
        2. audio.wav sr=16kHz , what script:
        - in the future
          - mfcc.csv
          - label.csv
        - ** be sure with channel for each audio you want to take data for signal

* go to each signal.csv
    - if the last chunk has size smaller than (16000, 1)
        - then delete the folder


* make mfcc
    - make_mfcc.py
    - make_mfcc_for_unknow_folder.py # was a problem so it need a spesific code

* check for same shape of mfcc.csv -> because we need it standart for the CNN
    - chech_shape_of_mfcc.py

* make mfcc and save it in the directory
* make the label.csv
-----------------------------------------------------------------
the above will create the FINISHED_V6 that contains for every recording :
    * signal.csv
    * mfcc.csv
    * label.csv