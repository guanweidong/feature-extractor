# Feature Extractor v2

* PCA features  
* Features from **onset classifier**

## Requirements
tensorflow==1.1.0   
youtube-dl==2017.5.14  
tqdm==4.11.2  
librosa==0.4.3   
numpy==1.12.1  
pandas==0.19.2   
bs4==0.0.1  

ffmpeg(installed by **apt-get or brew(Mac OSx)**)

## utils.py
Used to make several functions for ops.py

## ops.py

### query2wav(file_path)
1. Get query in file from **file_path**  
2. Use **add_music_link** to get links from query on soundcloud  
3. Sort the link and delete overlapped links  
4. Create wav_dir named same with file_path  
5. Download the music according to links  
6. Change mp3 file to wav file in wav_dir   

~~~
from ops import query2wav  
query2wav('./query.txt')
~~~

### music_load(music_path, start=0.0, end=60.0):
**librosa.load** wrapper for usage  

1. Sampling rate is fixed with SR(=22050)  
2. Load the music in **music_path** from **start**(sec) and **end**(sec)  
3. Return y(1D array), sr(=SR)  

~~~
from ops import music_load
y, sr = music_load('./test_sounds/test.wav', start = 1.0, end = 3.0)  
~~~

### pca_generator(clip_sec = 0.1, wav_path=WAV_PATH)

1. Load every sounds in **wav_path**  
2. Convert sounds clipped with **clip_sec** to spectogram  
3. Stack every spectogram and get covariance matrix  
4. Get pca basis vectors and store in **'./pca_dataset/'**  
5. **pca_basis_(clip_sec).pkl** is generated  
~~~
from ops import pca_generator  
pca_generator()  
~~~

### get_features(y, tss = 0.1, npca_comp = 30)
**tss** : time segment size(sec)   
**npca_comp** : number of pca components  
**tseg_len** : int(**tss** $\times$ SR) length of time segment  
**ntseg** : len(y)//**tseg_len**  number of time segment  

1. **get_pca_features** to get pca features   
2. **get_reduced_onset_features** to get onset features  
3. Concatenate to make **features**  
4. Generate **header** to contain every features information

~~~
from ops import get_features
header, features = get_features(y, tss =0.1, npca_comp = 30)
~~~

### music2csv(music_path, csv_path = './features/features.csv', start = 0, end = 1, tss = 0.1, npca_comp = 30)

1. Load the music in **music_path** with **music_load** from **start**(sec) to **end**(sec)  
2. Get header and features from **get_features**   
3. Generate index according to the **tss**  
4. Store features with header_info on **csv_path**   

~~~
from ops import music2csv   
music2csv(music_path = './test_sounds/test.wav',  
          csv_path = './features/features.csv',  
          start = '0.0',
          end = '3.0',  
          tss = 0.1,  
          npca_comp = 30)  
~~~

## Recommend usage

1. Add words on "query.txt"

2. **Download wav files**

~~~
python main.py --download
~~~

3. **Extract features from  wav files**

~~~
python main.py --feature
~~~
