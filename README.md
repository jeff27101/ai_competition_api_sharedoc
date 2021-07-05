# Esun 2021 ai_competition_api_sharedoc

## Contents
1. [About Competition & Requirements](#1-about-competition-requirements)
2. [System](#2-system)
3. [Inference](#3-inference)
4. [Performance Benchmark (web service)](#4-benchmark) 
5. [Summary](#5-summary)

## 1. About Competition Requirements

### 1.1 About Competition
本次比賽除了建構和訓練模型外，需要將模型部署於API Server，讓這項挑戰更貼近於現實生活。
依照主辦方說明：正式賽期間將發送HTTP request(POST)提供題目(Inference)，參賽者API回傳模型運算結果，timeout時間為1秒.

### 1.2 Requirements
#### -1 Instance Spec & Estimation
主辦要求: **response time less than 1 sec**

考量到response time，本次競賽一開始就打算以小型模型為目標，記憶體需求其實不高，預計Ensemble 3-4個模型約150MB，加上Backend Service以及其他系統單元，總共大約600MB(起動兩組web service)，
評估下來 1GB RAM的機器已綽綽有餘。

| Model | Model Parameters | *RAM Consumption |
| -------- | -------- | -------- |
| MobileNetV3-Large-0.75 | 3.7 M | 32.46 MB |
| MobileNetV3-Large-1.0 | 5.2 M | 39.70 MB |
| EfficientNet-Lite-2 | 5.8 M | 30.15 MB |

***RAM Consumption** is mesuered by calculating system RAM elimation after ONNX runtime loading the model 


#### -2 Cloud Platform Comparison 

優化成本應該是考量真實產品服務中很重要的一環，期望能在滿足需求的條件下，以最便宜的價格運作服務，以期達到每單位使用者數最低的花費成本。
比較了主流的雲端服務平台AWS, GCP與相較較便宜的DigitalOcean(DO)，其中在不考慮*Spot的機器下，DO有明顯的價格優勢，此外DO機器設定簡單上許多

***Spot**: 雲端服務商有權隨時收回機器，以換取相較便宜許多的機器價格。但要是剛好在比賽過程中被收回會很麻煩


| Platform | Type & Spec | Price | Price Factor |
| -------- | -------- | -------- | -------- |
| DO | s-1vcpu-1gb (1vCPU 1GB RAM) | $5/mo | X1 |
| AWS | t4g.micro (2vCPU 1GB RAM) | $7.776/mo | X1.555 |
| GCP | e2-micro (0.25vCPU 1GB RAM) | $7.84/mo | X1.568 |


## 2. System

### 2.1 Architecture
![](https://i.imgur.com/VbHlGQa.png)

### 2.2 Build and Deployment
相較於訓練模型的程式碼，Web Service程式碼要盡可能穩定、隔離、版本好管理、好部署，還要方便在Local測試，因此我使用Docker來打包整個服務。
打包部署流程如下：

1. 下載訓練完的模型們
2. 建立Docker image
3. 推上Docker registry
4. 連線進雲端機器
5. 拉下最新Docker image
6. 啟動容器
7. 測試API是否正常

為了簡化指令，將指令做成Makefile，這點考量的是避免輸入錯誤指令的災難後果。

``` shell
# 進到容器
$ make build && make devenv

# 在Local起服務
$ make build && make run

# 打包並推上registry
$ make build && make push
```

### 2.3 API Document
![](https://i.imgur.com/ULWE3xP.png)

這次挑戰使用FastAPI來寫網頁服務，FastAPI號稱是當代最快的python網頁服務。除了速度快之外，FastAPI包含一些好用的工具，像是內建支援OpenAPI,  JSON Schema和Pydantic，減少了寫驗證程式碼的時間，也讓程式碼可讀性大為增加。
定義好function後，只需要用瀏覽器打開http://localhost:8080/docs 即可看到API文件

![](https://i.imgur.com/8VHtDi8.png)

## 3. Inference

### 3.1 ONNX Runtime

![](https://i.imgur.com/X134taI.png)

比賽初期，會使用Tensorflow或Pytorch來訓練Model，如果Web Service把TF/Pytorch framework都打包進去那Image會變得非常龐大。此外Model checkpoint和Model本身含有很多跟training有關的變數，增加不少Model size.
使用ONNX Runtime就是來解決這個問題，使用統一的ONNX格式就不用擔心訓練時用不同的framework會需要改inference code，而且ONNX Runtime本身檔案非常小(才5MB左右)，TF和Pytorch model輸出成onnx格式會去除不必要的op和variable，Model檔案會縮小至原本的1/3左右(45.2MB -> 14.9MB)，讓docker image檔案盡量的小，減少部署時pull image時間。

### 3.1 Ensemble Model

![](https://i.imgur.com/cX0SSjI.png)

其實Ensemble model是比賽進行到Day3才決定要上線的，原先是挑一個表現最好的model上線，但在收集和觀察前兩天跑的結果後，發現到模型對於某些文字輸出不是非常肯定(Low probability)，另外手上每個model擅長的字都不太一樣，因此決定用Ensemble的方式。
比較過幾種Ensemble方法，最後決定直接取平均似乎是一個公平穩定的做法，Ensemble使用下列三個模型：MobleNetV3-Large-0.75, MobileNetV3-Large-1, EfficientNet-Lite-2，讓他們互補彼此弱項，發揮所長。最後也有反映在成績上的成長。

###### Example
本題文字就出現model意見分歧狀況，兩個模型覺得是勇，一個模型覺得是男不過沒這麼肯定，整體來說"勇"的平均機率比"男"高，最後ensemble結果為"勇"

![](https://i.imgur.com/jyZLK9Q.png)



## 4. Benchmark

這部分為比賽結束後才做，測試賽的時候確認過Web Service沒問題就先專心在訓練模型上，但因為真實世界負載是非常重要的，需要知道服務的極限，才能確保服務能穩定服務使用者。

### 4.1 極限測試
這個在測試服務的極限在哪，模擬100使用者以間隔2-5秒的速度送出request
得到9.6RPS，即每秒能處理9.6個字
不過可以看到Response time已經超過主辦單位要求的1秒內回應，所以實際上不能達到這個量

![](https://i.imgur.com/fCcHXK6.png)

### 4.2 負荷測試
測試了幾個參數，大致上這個參數能滿足主辦單位的要求，為可被接受的負荷量
模擬70使用者以間隔2-5秒的速度送出request
得到約6.6RPS，即每秒能處理6.6個字
假若知道未來產品的負荷量，除以這數字就知道要開多少台機器，以及成本多少了

![](https://i.imgur.com/hdlFk1Y.png)

## 5. Summary
首先要感謝主辦單位和T-Brain，參加比賽是寶貴的經驗
分數上最後總排名#21，從Day1~Day4每天的當日排名都有持續進步
還可以再好好想想許多方法來作嘗試

#### Team: "DeepBlueShark"
|  | Daily Score | Daily Ranking | Overall Score| Overall Ranking|
| -------- | -------- | -------- | ------- | -------- |
| Day1 | 0.86589 | #39 | 0.86589 | #39 |
| Day2 | 0.92286 | #23(↑16) | 0.89828 | #32(↑7) |
| Day3 | 0.95330 | #11(↑12) | 0.91748 | #26(↑6) | 
| Day4 | 0.96370 | #10(↑1) | 0.92969 | #21(↑5) |


