# zoiとは
zoiとは4コマ漫画の画像からコマを抽出するツール群です。
簡単な画像処理によって実現される予定です。

### 依存関係
* Python 2.x
* OpenCV 2.x
* Python Imaging Library ( Pillow )
* pathlib
* numpy
* matplotlib

Python 3.xで開発するつもりだったのですが、何故か動かなかったのでとりあえず2.xで.

### 各ファイルの解説
#### cut.py
メインで切り出しを行います。  

### filter.py
ヒストグラム正規化をして、濃淡をいい感じに補正します。Scansnap対応。

### noisereduc.py
名前のとおり

### やること
- [x] 切り出し
- [x] 画像をきれいに
- [ ] 傾き検出
- [ ] 傾き自動補正
- [ ] 完全自動化
