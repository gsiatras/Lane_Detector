from lane_detector import LaneDetector
import cv2


def setup_input_1frame():
    image_path = './img_321.jpg'
    frame = cv2.imread(image_path)
    return frame


def setup_camera_input():
    """
    get live camera input
    :return:
    """



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    frame = setup_input_1frame()

    #cam = setup_camera_input()

    ld = LaneDetector()

    #ld.process_video(cam)

    lanes = ld.detect_lanes(frame)

    angle_frame = ld.detect_steering_angle(frame, lanes)

    ld.show_image("DSAD", angle_frame)


    #lane_lines_image = ld.display_lines(frame, lanes)








   # cv2.imshow("lane lines", lane_lines_image)





