#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

using namespace cv;
using namespace std;

const cv::String photoPath = R"(pics\hot-charcoal.jpg)";

void wave(const Mat &image, Mat &result) {
    Mat srcX(image.rows, image.cols, CV_32F);
    Mat srcY(image.rows, image.cols, CV_32F);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            srcX.at<float>(i, j) = j;
            srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
        }
    }

    remap(image,
          result,
          srcX,
          srcY,
          INTER_LINEAR
    );
}

void sharpen(const Mat &image, Mat &result) {
    result.create(image.size(), image.type());
    int nChannels = image.channels();
    for (int j = 1; j < image.rows - 1; j++) {
        const auto *previous = image.ptr<const uchar>(j - 1);
        const auto *current = image.ptr<const uchar>(j);
        const auto *next = image.ptr<const uchar>(j + 1);
        auto *output = result.ptr<uchar>(j);
        for (int i = nChannels; i < (image.cols - 1) * nChannels; i++) {
            *output++ = saturate_cast<uchar>(
                    5 * current[i] - current[i - nChannels] -
                    current[i + nChannels] - previous[i] - next[i]);
        }
    }
    result.row(0).setTo(Scalar(1));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}

void colorReduceb(Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols * image.channels();
    if (image.isContinuous()) {
        nc = nc * nl;
        nl = 1;
    }
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    uchar mask = 0xFF << n;
    uchar div2 = div >> 1;
    for (int j = 0; j < nl; j++) {
        auto *data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            *data &= mask;
            *data++ += div2;
        }
    }
}

void colorReduce(Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols * image.channels();
    for (int j = 0; j < nl; j++) {
        auto *data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            data[i] = data[i] / div * div + div / 2;
        }
    }
}


void colorReducei(Mat image, int div = 64) {
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    uchar mask = 0xFF << n;
    uchar div2 = div >> 1;

    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itend = image.end<Vec3b>();
    for (; it != itend; ++it) {
        (*it)[0] &= mask;
        (*it)[0] += div2;
        (*it)[1] &= mask;
        (*it)[1] += div2;
        (*it)[2] &= mask;
        (*it)[2] += div2;
    }
}

void salt(Mat image, int n) {
    default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, image.rows - 1);
    uniform_int_distribution<int> randomCol(0, image.cols - 1);

    int i, j;
    for (int k = 0; k < n; k++) {
        i = randomCol(generator);
        j = randomRow(generator);
        if (image.type() == CV_8UC1) {
            image.at<uchar>(j, i) = 255;
        } else if (image.type() == CV_8UC3) {
            image.at<Vec3b>(j, i) = Vec3b(255, 255, 255);
        }
    }
}

void pepper(Mat image, int n) {
    default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, image.rows - 1);
    uniform_int_distribution<int> randomCol(0, image.cols - 1);

    int i, j;
    for (int k = 0; k < n; k++) {
        i = randomCol(generator);
        j = randomRow(generator);
        if (image.type() == CV_8UC1) {
            image.at<uchar>(j, i) = 255;
        } else if (image.type() == CV_8UC3) {
            image.at<Vec3b>(j, i) = Vec3b(0, 0, 0);
        }
    }
}

void saltBlue(Mat image, int n) {
    default_random_engine generator;
    uniform_int_distribution<int> randomRow(0, image.rows - 1);
    uniform_int_distribution<int> randomCol(0, image.cols - 1);

    int i, j;
    for (int k = 0; k < n; k++) {
        i = randomCol(generator);
        j = randomRow(generator);
        if (image.type() == CV_8UC1) {
            image.at<uchar>(j, i) = 255;
        } else if (image.type() == CV_8UC3) {
            image.at<Vec3b>(j, i) = Vec3b(255, 0, 0);
        }
    }
}

int main() {
    Mat result;

    // 1. Відображення оригінального зображення
    Mat image = imread(photoPath);
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", image);

    // 2. Додавання солі
    Mat imageSalt = image.clone();
    salt(imageSalt, 3000);
    namedWindow("Salt", WINDOW_AUTOSIZE);
    imshow("Salt", imageSalt);

    // 3. Додавання перцю
    Mat imagePepper = image.clone();
    pepper(imagePepper, 3000);
    namedWindow("Pepper", WINDOW_AUTOSIZE);
    imshow("Pepper", imagePepper);

    // 4. Зменшення кількості кольорів в 64 рази
    Mat imageColorReduce = image.clone();
    const int64 start = getTickCount();
    colorReduce(imageColorReduce);
    double durationSimple = (getTickCount() - start) / getTickFrequency();
    cout << "Speed (Color Reduce, 64 times) =" << durationSimple << endl;
    namedWindow("Color Reduce (64t)", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t)", imageColorReduce);

    // 5. Зменшення кількості кольорів в 64 рази, за допомогою ітераторів
    imageColorReduce = image.clone();
    const int64 starti = getTickCount();
    colorReduce(imageColorReduce);
    double durationIterators = (getTickCount() - starti) / getTickFrequency();
    cout << "Speed (Color Reduce using Iterators, 64 times) = " << durationIterators << endl;
    namedWindow("Color Reduce (64t): Iterators", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t): Iterators", imageColorReduce);

    // 6. Тест безперервного зображення
    Mat image_B = image.clone();
    colorReduceb(image_B, 64);
    const int64 startb = getTickCount();
    colorReduce(image_B);
    double durationb = (getTickCount() - startb) / getTickFrequency();
    cout << "speed b =" << durationb << endl;
    namedWindow("image B", WINDOW_AUTOSIZE);
    imshow("image B", image_B);

    // 7. Сканування зображення з сусіднім доступом та зменшення
    // кількості кольорів  у кожному каналі у 64
    Mat images = image.clone();
    const int64 starts = getTickCount();
    colorReduce(images);
    double durations = (getTickCount() - starts) / getTickFrequency();
    cout << "speed sharpen = " << durations << endl;
    namedWindow("Image Sharpen", WINDOW_AUTOSIZE);
    imshow("Image Sharpen", images);

    // 8.	Покращимо чорно-біле зображення
    Mat image64grey = image.clone();
    colorReduce(image64grey, 64);
    namedWindow("Image 64 grey", WINDOW_AUTOSIZE);
    imshow("Image 64 grey", image64grey);

    sharpen(image64grey, result);
    namedWindow("Image sharpen grey", WINDOW_AUTOSIZE);
    imshow("Image sharpen grey", result);

    // 9. Поєднання двух зображень за допомогою арифметичного оператора
    sharpen(image, result);
    namedWindow("Image op", WINDOW_AUTOSIZE);
    imshow("Image op", result);

    // 10. Эффект дощ
    Mat imageBlue = image.clone();
    saltBlue(imageBlue, 3000);
    namedWindow("Image Blue", WINDOW_AUTOSIZE);
    imshow("Image Blue", imageBlue);

    // 11. Створення хвилеподібного еффекту
    wave(image, result);
    namedWindow("Image Wave", WINDOW_AUTOSIZE);
    imshow("Image Wave", result);

    waitKey(0);
    return 0;
}
