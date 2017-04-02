#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QDialog>
#include <QMainWindow>
#include<opencv2/videoio.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
using namespace cv;
namespace Ui {
class MainWindow;
}
class MainWindow : public QMainWindow
{
    Q_OBJECT
public slots:
    void processFrameAndUpdateGUI();
    void post_data(int counterNumber, int counterCount, double avgWaitingTime, double throughput);
    cv::Mat detectAndDraw_people(const HOGDescriptor &hog, Mat &img , CascadeClassifier& cascade,
                                          CascadeClassifier& nestedCascade,
                                          double scale, bool tryflip);





public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:

    //Slot for the load video push button.
    void on_pushButton_clicked();
    // Slot for the play push button.
    void on_pushButton_2_clicked();
private:

    cv::VideoCapture capWebcam;
    cv::Mat matOriginal;
    cv::Mat matProcessed;
    //void post_data();

    QImage qimgOriginal;
    QImage qimgProcessed;
    int CounterCount,currentIndex;
    int counters[5];
    std::vector<cv::Vec3f> vecCircles;
    std::vector<cv::Vec3f>::iterator itrCircles;

    QTimer* qtimer;
    QTimer* qtimer2;
    Ui::MainWindow *ui;

};
#endif // MAINWINDOW_H
