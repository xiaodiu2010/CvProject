### To test this project in following steps:

1.Run the demo.m in /pose. And get the human pose estimation image in /database/poseresult/.
you can skip this step ,I have output a demo in /database/poseresult/.

2.When first run this project, you need to train unary model.Run:

```
python preprocessing.py -p
# -p means use pretrain model
```    

This will take some time.

3.Run the mrf,Run:

`python mrf.py`

you would see the mrfresult in ./database/mrfresult/



### To Run the whole project from scratch:

1.Download all the dataset.

In project dictionary ,run:

```
chmod +x getdata.sh
./getdata.sh
```

This will download data from dataset and combine into database dictionary

2.Run the demo.m in /pose. And get the human pose estimation image in /database/poseresult/.


3.processing all data,Run: 

`python preprocessing.py`

4.Run the mrf,Run:

`python mrf.py`

you would see the mrfresult in ./database/mrfresult/



other information.
the pygco need Compile if you download the source file from github.
This is python wrapper for gco-3.0 package. you can follow the instruction of github: https://github.com/yujiali/pygco




