#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <random>

using namespace cv;
using namespace std;

const cv::String photoPath = R"(pics\hot-charcoal.jpg)";
const cv::String patternPath = R"(pics\2.jpg)";

void wave(const Mat &image, Mat &result) {
    result.create(image.size(), image.type());

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

void addImage(const Mat &source, const Mat &apply, Mat &result) {
    result.create(source.size(), source.type());

    Mat applyWork;
    applyWork.create(source.size(), source.type());
    resize(apply, applyWork, Size(source.cols, source.rows), INTER_LINEAR);
    normalize(applyWork, applyWork, 0 , 1, NORM_MINMAX);

    multiply(source, applyWork, result);
}

void sharpen(const Mat &image, Mat &result) {
    result.create(image.size(), image.type());
    int nChannels = image.channels();

    for (int j = 1; j < image.rows - 1; j++) {
        // rows
        const auto *previous = image.ptr<const uchar>(j - 1);
        const auto *current = image.ptr<const uchar>(j);
        const auto *next = image.ptr<const uchar>(j + 1);
        auto *output = result.ptr<uchar>(j);
        
        // sharpening
        int total = (image.cols - 1) * nChannels;
        for (int i = nChannels; i < total; i++) {
            uchar cen = current[i];
            uchar left = current[i - nChannels];
            uchar right = current[i + nChannels];
            uchar up = previous[i];
            uchar down = next[i];
            // frac result and write to output row
            *output++ = saturate_cast<uchar>(
                    5 * cen - left - right - up - down
            );
        }
    }
    
    result.row(0).setTo(Scalar(1));
    result.row(result.rows - 1).setTo(Scalar(0));
    result.col(0).setTo(Scalar(0));
    result.col(result.cols - 1).setTo(Scalar(0));
}

// Реалізація з використанням ядра фільтру
void sharpenKernel(const cv::Mat &image,cv::Mat &result) {
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));

    kernel.at<float>(1,1)= 5.0;
    kernel.at<float>(0,1)= -1.0;
    kernel.at<float>(2,1)= -1.0;
    kernel.at<float>(1,0)= -1.0;
    kernel.at<float>(1,2)= -1.0;

    cv::filter2D(image,result,image.depth(),kernel);
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

// Реалізація з використання побітових операцій
void colorReduceContinuousByte(Mat image, int div = 64) {
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
            *data++ |= div2;
        }
    }
}

void colorReduceContinuous(Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols * image.channels();

    if (image.isContinuous()) {
        nc = nc * nl;
        nl = 1;
    }

    for (int j = 0; j < nl; j++) {
        auto *data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            data[i] = data[i] / div * div + div / 2;
        }
    }
}

void colorReduceAt(Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols;
    for (int j = 0; j < nl; j++) {
        for (int i = 0; i < nc; i++) {
            auto &vec = image.at<cv::Vec3b>(j, i);
            for(int c = 0; c < 3; c++) {
                vec[c] = vec[c] / div * div + div / 2;
            }
        }
    }
}

void colorReduceIter(Mat image, int div = 64) {
    Mat_<Vec3b>::iterator it = image.begin<Vec3b>();
    Mat_<Vec3b>::iterator itEnd = image.end<Vec3b>();

    while (it != itEnd) {
        Vec3b &pixel = *it;
        for (int i = 0; i < 3; i++) {
            pixel[i] = pixel[i] / div * div + div / 2;
        }
        ++it;
    }
}

void colorReduce(Mat image, int div = 64) {
    int nl = image.rows;
    int nc = image.cols * image.channels();
    for (int j = 0; j < nl; j++) {
        // Адреса рядка даних j
        auto *data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++) {
            // Оновлення пікселів
            data[i] = data[i] / div * div + div / 2;
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
    pepper(imagePepper, 20000);
    namedWindow("Pepper", WINDOW_AUTOSIZE);
    imshow("Pepper", imagePepper);

    // 4. Зменшення кількості кольорів в 64 рази, за допомогою показчиків
    Mat imageColorReduce = image.clone();
    const int64 startSimple = getTickCount();
    colorReduce(imageColorReduce);
    double durationSimple = (getTickCount() - startSimple) / getTickFrequency();
    cout << "Speed (Color Reduce, 64 times) = " << durationSimple << endl;
    namedWindow("Color Reduce (64t)", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t)", imageColorReduce);

    // 5. Зменшення кількості кольорів в 64 рази, за допомогою ітераторів
    imageColorReduce = image.clone();
    const int64 startIter = getTickCount();
    colorReduceIter(imageColorReduce);
    double durationIter = (getTickCount() - startIter) / getTickFrequency();
    cout << "Speed (Color Reduce using Iterators, 64 times) = " << durationIter << endl;
    namedWindow("Color Reduce (64t): Iterators", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t): Iterators", imageColorReduce);

    // 6. Перевірити процес сканування тестового зображення з
    // використанням методу at для доступу до пікселів
    imageColorReduce = image.clone();
    const int64 startAt = getTickCount();
    colorReduceAt(imageColorReduce);
    double durationAt = (getTickCount() - startAt) / getTickFrequency();
    cout << "Speed (Color Reduce using at(), 64 times) = " << durationAt << endl;
    namedWindow("Color Reduce (64t): at()", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t): at()", imageColorReduce);

    // 7. Тест безперервного зображення
    imageColorReduce = image.clone();
    const int64 startCont = getTickCount();
    colorReduceContinuous(imageColorReduce, 64);
    double durationCont = (getTickCount() - startCont) / getTickFrequency();
    cout << "Speed (Color Reduce Continuous, 64 times) = " << durationCont << endl;
    namedWindow("Color Reduce (64t): Continuous", WINDOW_AUTOSIZE);
    imshow("Color Reduce (64t): Continuous", imageColorReduce);

    // 8. Сканування зображення з сусіднім доступом та зменшення
    // кількості кольорів  у кожному каналі у 64
    const int64 starts = getTickCount();
    sharpen(image, result);
    double durations = (getTickCount() - starts) / getTickFrequency();
    cout << "Speed (Sharpen) = " << durations << endl;
    namedWindow("Image Sharpen", WINDOW_AUTOSIZE);
    imshow("Image Sharpen", result);

    // 9. Покращити різкість чорно-білої версії тестового
    // зображення щонайменше двома способами та порівняти результат.
    Mat imageGray = image.clone();
    cvtColor(imageGray, imageGray, COLOR_BGR2GRAY);
    namedWindow("Image Gray", WINDOW_AUTOSIZE);
    imshow("Image Gray", imageGray);

    sharpen(imageGray, result);
    namedWindow("Image Sharpen Gray", WINDOW_AUTOSIZE);
    imshow("Image Sharpen Gray", result);

    // 10. Поєднайте два зображення за допомогою арифметичного оператора
    const Mat &patternImage = imread(patternPath);
    addImage(image, patternImage, result);
    namedWindow("Image op", WINDOW_AUTOSIZE);
    imshow("Image op", result);

    // 11. Додайте «дощ» до синього каналу тестового зображення
    Mat imageBlue = image.clone();
    saltBlue(imageBlue, 3000);
    namedWindow("Image Blue", WINDOW_AUTOSIZE);
    imshow("Image Blue", imageBlue);

    // 12. Створіть хвилеподібний ефект на зображенні
    wave(image, result);
    namedWindow("Image Wave", WINDOW_AUTOSIZE);
    imshow("Image Wave", result);

    waitKey(0);
    return 0;
}
