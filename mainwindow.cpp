#include "mainwindow.h"
#include "ui_mainwindow.h"
#include<opencv2/videoio.hpp>
#include<opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include<iostream>
#include<string.h>
#include<QtCore>
#include<QNetworkRequest>
#include<QUrl>
#include<QtNetwork>
#include <QtNetwork/QNetworkAccessManager>
#include<QObject>
#include<QString>
using namespace std;
using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //int count =0;
    //post_data();
    int cam_feed = ui->comboBox->currentIndex();
   // capWebcam.open(cam_feed-1);
    capWebcam.open(0);
    currentIndex=cam_feed;
    if(capWebcam.isOpened() == false) {
        //ui->txtXYRadius->appendPlainText("error: capWebcam not accessed successfully");
        return;
    }


       qtimer = new QTimer(this);
        connect(qtimer, SIGNAL(timeout()), this, SLOT(processFrameAndUpdateGUI()));
        qtimer->start(20);


//    qtimer2 = new QTimer(this);
//    connect(qtimer2,SIGNAL(timeout()),this, SLOT(post_data()));
//    qtimer->start(10000);
}

void MainWindow::post_data(int counterNumber, int counterCount, double avgWaitingTime, double throughput)
{
        QUrl serviceUrl = QUrl("https://througputcalc.herokuapp.com/api/updateCounter");
        QByteArray postData;
        postData.append("airportName=Indira Gandhi International Airport&");
        postData.append("carrierName=Jet Airways&");
//        postData.append("source=Kolkata&");
//        postData.append("destination=Delhi&");
//        postData.append("delayed=600000&");
//        postData.append("boardingGate=21A&");
//        postData.append("departureTime=1490994130000&");
//        postData.append("arrivalTime=1491004930000&");
//        postData.append("flightNo=IX-202");

            QString s1 = "counterNumber=" + QString::number(counterNumber) + "&";
            QString s2 = "counterCount=" + QString::number(counterCount) + "&";
            QString s3 = "avgWaitingTime=" + QString::number(avgWaitingTime) + "&";
            QString s4 = "throughput=" + QString::number(throughput);

            postData.append(s1);
            postData.append(s2);
            postData.append(s3);
            postData.append(s4);
        QString a = "POSTED: IGI Airport , Jet Airways , Counter Number:" + QString::number(counterNumber) + " WT: "+ QString::number(avgWaitingTime)+ " Count: "+ QString::number(counterCount);
           ui->label_13->setText(a);
        QNetworkAccessManager* networkManager = new QNetworkAccessManager;
        QObject::connect(networkManager, SIGNAL(finished(QNetworkReply*)), SLOT(serviceRequestFinished(QNetworkReply*)));
        networkManager->post(QNetworkRequest(serviceUrl), postData);
}

void MainWindow::processFrameAndUpdateGUI()
{
    if(currentIndex!= ui->comboBox->currentIndex())
         {
        currentIndex= ui->comboBox->currentIndex();
        if(currentIndex==0)
            capWebcam.open(0);
        else if(currentIndex==1)
        capWebcam.open("/home/cooper/Downloads/1.MP4");
        else if(currentIndex==2)
            capWebcam.open("/home/cooper/Downloads/2.MP4");
        else if(currentIndex==3)
                    capWebcam.open("/home/cooper/Downloads/3.MP4");
    }
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    //double scale;
    VideoCapture vc;
    Mat frame,image;
    cascade.load("/home/cooper/sih/data/haarcascades/haarcascade_frontalface_alt.xml");
    nestedCascade.load("/home/cooper/sih/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");

    capWebcam.read(matOriginal);

    if(matOriginal.empty() == true) return;

    //START
    matProcessed = matOriginal.clone();
    matProcessed = detectAndDraw_people(hog,matProcessed,cascade, nestedCascade,1, tryflip );
    //END

   // cv::cvtColor(matOriginal, matOriginal, CV_BGR2RGB);
    QImage qimgOriginal((uchar*)matOriginal.data, matOriginal.cols, matOriginal.rows, matOriginal.step, QImage::Format_RGB888);
    QImage qimgProcessed((uchar*)matProcessed.data, matProcessed.cols, matOriginal.rows, matProcessed.step, QImage::Format_RGB888);

    ui->lblOriginal->setPixmap(QPixmap::fromImage(qimgOriginal));
    ui->lblProcessed->setPixmap(QPixmap::fromImage(qimgProcessed));
}

Mat MainWindow::detectAndDraw_people(const HOGDescriptor &hog, Mat &img , CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip)
{
    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };
    Mat gray, smallImg;

    cvtColor( img, gray, COLOR_BGR2GRAY );
    //smallImg = img.clone();
    double fx = 1 / scale;
    cv::resize( gray, smallImg, Size(), fx, fx, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );

    t = (double)getTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        //|CASCADE_FIND_BIGGEST_OBJECT
        //|CASCADE_DO_ROUGH_SEARCH
        |CASCADE_SCALE_IMAGE,
        Size(30, 30) );
    if( tryflip )
    {
        flip(smallImg, smallImg, 1);
        cascade.detectMultiScale( smallImg, faces2,
                                 1.1, 2, 0
                                 //|CASCADE_FIND_BIGGEST_OBJECT
                                 //|CASCADE_DO_ROUGH_SEARCH
                                 |CASCADE_SCALE_IMAGE,
                                 Size(30, 30) );
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    std::cout<<"Face's Detected"<<faces.size()<<endl;

    vector<Rect> found, found_filtered;
    double t1 = (double) getTickCount();
    // Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    t1 = (double) getTickCount() - t1;
    cout << "detection time = " << (t1*1000./cv::getTickFrequency()) << " ms" << endl;
    for(size_t i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];

        size_t j;
        // Do not add small detections inside a bigger detection.
        for ( j = 0; j < found.size(); j++ )
            if ( j != i && (r & found[j]) == r )
                break;

        if ( j == found.size() )
            found_filtered.push_back(r);
    }
    cout<<"count"<<found_filtered.size()<<endl;
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cvPoint(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        nestedCascade.detectMultiScale( smallImgROI, nestedObjects,
            1.1, 2, 0
            //|CASCADE_FIND_BIGGEST_OBJECT
            //|CASCADE_DO_ROUGH_SEARCH
            //|CASCADE_DO_CANNY_PRUNING
            |CASCADE_SCALE_IMAGE,
            Size(30, 30) );
        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    for (size_t i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];

        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
    }
    int total = (found_filtered.size() + faces.size());
    CounterCount = total;
    int counterNumber = qrand();
    counterNumber = counterNumber%5 + 1;

      int throughput = qrand();
    throughput = throughput%5;
    if(throughput<3)
        throughput += 3;
    int dec = qrand(),tmp=qrand();
    dec%=10;tmp%=10;
    double avgWaitingTime = CounterCount*throughput;
    counters[counterNumber]=avgWaitingTime;
    int dest=counters[counterNumber];
    int f =0;
    for(int i=1;i<5;i++)
    {
        if(counters[i]<dest)
        {dest=counters[i];
            f=i;}
    }
    ui->label_12->setText(QString::number(f+1));
      ui->label_10->setText(QString::number(throughput) + "." + QString::number(dec));
      dec*=throughput;
      dec/=10;
    ui->label_9->setText(QString::number(avgWaitingTime + dec) + "." + QString::number(tmp));

    ui->label_11->setText(QString::number(total));
    return img;
}


MainWindow::~MainWindow()
{

    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
  int counterNumber = qrand();
  counterNumber = counterNumber%4 + 1;
  //int counterCount = qrand()
    int throughput = qrand();
  throughput = throughput%5;
  if(throughput<3)
      throughput += 3;
  double avgWaitingTime = CounterCount*throughput;
post_data(counterNumber,CounterCount,avgWaitingTime,throughput);

}
void MainWindow::on_pushButton_2_clicked()
{
    if(qtimer->isActive() == true) {
        qtimer->stop();
        ui->pushButton_2->setText("Resume");
    } else {
        qtimer->start(20);
        ui->pushButton_2->setText("Pause");
    }
}
