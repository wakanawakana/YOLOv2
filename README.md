# YOLOv2
https://github.com/leetenki/YOLOv2

使用した元のchnainer版 YOLOv2

## 確認環境
- windows 7 64bit
- Anaconda 4.0.0
- Python 2.7
- OpenCV 2.X
- Chainer 1.16.0

## YOLOv2のgrad-cam検討
Grad-camによる可視化のための変更をかけています

これで動いてるのでしょうか？？

問題点
- 論文のImgeNetのようなクラス分類でないこと
- BBOXのbackpropagationをどうするのか（x,y,w,h）のLoss
- おそらく画面関係が対応しない20層以降
- 入力画像サイズで反応が違う（原因不明）

添付写真は18層の反応を女の人から作ったもの

<img src="data/people.png" width="280" height="280"><img src="data/yolov2_result.jpg"  width="280" height="280"><img src="data/gcam-18.png"  width="280" height="280"> 

以降22層から全部ならべてみた

<img src="data/gcam-22.png"  width="320" height="320">
<img src="data/gcam-21.png"  width="320" height="320">
<img src="data/gcam-20.png"  width="320" height="320">
<img src="data/gcam-19.png"  width="320" height="320">
<img src="data/gcam-18.png"  width="320" height="320">
<img src="data/gcam-17.png"  width="320" height="320">
<img src="data/gcam-16.png"  width="320" height="320">
<img src="data/gcam-15.png"  width="320" height="320">
<img src="data/gcam-14.png"  width="320" height="320">
<img src="data/gcam-13.png"  width="320" height="320">
<img src="data/gcam-12.png"  width="320" height="320">
<img src="data/gcam-11.png"  width="320" height="320">
<img src="data/gcam-10.png"  width="320" height="320">
<img src="data/gcam-9.png"  width="320" height="320">
<img src="data/gcam-8.png"  width="320" height="320">
<img src="data/gcam-7.png"  width="320" height="320">
<img src="data/gcam-6.png"  width="320" height="320">
<img src="data/gcam-5.png"  width="320" height="320">
<img src="data/gcam-4.png"  width="320" height="320">
<img src="data/gcam-3.png"  width="320" height="320">
<img src="data/gcam-2.png"  width="320" height="320">
<img src="data/gcam-1.png"  width="320" height="320">

grad-camの論文はこちら

https://arxiv.org/abs/1610.02391

論文のオリジナル実装（Lua）：https://github.com/ramprs/grad-cam

Keras：https://github.com/jacobgil/keras-grad-cam

Chainer：https://github.com/tsurumeso/chainer-grad-cam
