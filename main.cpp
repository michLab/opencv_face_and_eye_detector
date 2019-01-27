#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main()
{
    // Load an cascadeclassifier xml file:
    string face_cascade_name = "haarcascade_frontalface_default.xml";
    string eye_cascade_name = "haarcascade_eye.xml";
    CascadeClassifier face_cascade;
    CascadeClassifier eye_cascade;
    face_cascade.load(face_cascade_name);
    eye_cascade.load(eye_cascade_name);

    // Load an image and change it to gray:
    Mat image;// = imread("Hillary.jpg");
    VideoCapture cap;

    if (!cap.open(0))
    {
        cout << "Cannot open the video stream!" << endl;
    }

    Mat gray;

    while(true)
    {
        cap.read(image);
        //image = imread("Hillary.jpg");
        cvtColor(image, gray, COLOR_BGR2GRAY);
        //imshow("Image", image);
        //imshow("Image gray", gray);
        //waitKey(0);

        // Detect faces:
        vector<Rect> faces;
        face_cascade.detectMultiScale(gray, faces, 1.3, 5);

        if (faces.empty() == true)
        {
            cout << "No faces found";
            continue;
        }
        else
        {
            cout << faces.size() << " face(s) found" << endl;
        }

        // Detect eyes:
        vector<Rect> eyes;

        // Loop of detection:
        for (auto elem : faces)
        {
            Rect face_roi;
            face_roi.x = elem.x;
            face_roi.y = elem.y;
            face_roi.width = elem.width;
            face_roi.height = elem.height;
            rectangle(image, Point(face_roi.x,face_roi.y),
                      Point(face_roi.x+face_roi.width, face_roi.y+face_roi.height),
                      Scalar(127,0,255),2);
            Mat face_roi_color = image(face_roi);
            Mat face_roi_gray = gray(face_roi);

            eye_cascade.detectMultiScale(face_roi_gray, eyes);
            if (eyes.empty() == true)
            {
                cout << "No eyes found";
            }
            else
            {
                cout << eyes.size() << " eye(s) found" << endl;
            }
            for (auto elem_eye : eyes)
            {
                Rect eye_roi;
                eye_roi.x = elem_eye.x + face_roi.x;
                eye_roi.y = elem_eye.y + face_roi.y;
                eye_roi.width = elem_eye.width;
                eye_roi.height = elem_eye.height;

                rectangle(image, Point(eye_roi.x,eye_roi.y),
                          Point(eye_roi.x + eye_roi.width,
                                eye_roi.y + eye_roi.height),
                          Scalar(127,0,255),2);
                imshow("Face and eye detection", image);
            }
            ;
        }
        int k = waitKey(1);
        if (k == 27)
        {
           break;
        }
    }


    return 0;
}
