import cv2
import numpy as np


class LaneDetector(object):

    def __init__(self):
        pass


    def detect_edges(self, frame):
        """
        detect the edges of frame
        turn to hsv
        create a white mask and apply to get the edges
        :return: edges
        """
        # turn to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #self.show_image("hsv", hsv)

        # Define lower and upper thresholds for white color in HSV
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([255, 30, 255])

        # Create a mask to isolate white regions
        mask = cv2.inRange(hsv, lower_white, upper_white)

        #self.show_image("white mask", mask)

        # detect edges applying the Canny edge detection algorithm
        edges = cv2.Canny(mask, 200, 400)

        return edges



    def region_of_interest(self, edges):
        """
        keep only the edges in the down part of the image
        :param edges: detected edges
        :return: edges on the down part
        """
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # only focus bottom half of the screen
        polygon = np.array([[
            (0, height * 1 / 2),
            (width, height * 1 / 2),
            (width, height),
            (0, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        cropped_edges = cv2.bitwise_and(edges, mask)
        return cropped_edges


    def detect_line_segments(self, cropped_edges):
        """
        Perform Hough Line Transform to identify line segments in the input image
        tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
        :param cropped_edges:
        :return: line_segments
        """

        rho = 1  # distance precision in pixel, i.e. 1 pixel
        angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
        min_threshold = 10  # minimal of votes
        line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                        np.array([]), minLineLength=8, maxLineGap=4)

        return line_segments


    def average_slope_intercept(self, frame, line_segments):
        """
        This function combines line segments into one or two lane lines
        If all line slopes are < 0: then we only have detected left lane
        If all line slopes are > 0: then we only have detected right lane
        """
        lane_lines = []
        if line_segments is None:
            print('No line_segment segments detected')
            return lane_lines

        height, width, _ = frame.shape
        left_fit = []
        right_fit = []

        boundary = 1 / 3
        left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
        right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

        for line_segment in line_segments:
            for x1, y1, x2, y2 in line_segment:
                if x1 == x2:
                    print('skipping vertical line segment (slope=inf): %s' % line_segment)
                    continue
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:
                    if x1 < left_region_boundary and x2 < left_region_boundary:
                        left_fit.append((slope, intercept))
                else:
                    if x1 > right_region_boundary and x2 > right_region_boundary:
                        right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        if len(left_fit) > 0:
            lane_lines.append(self.make_points(frame, left_fit_average))

        right_fit_average = np.average(right_fit, axis=0)
        if len(right_fit) > 0:
            lane_lines.append(self.make_points(frame, right_fit_average))

        print('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

        return lane_lines


    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height  # bottom of the frame
        y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

        # bound the coordinates within the frame
        x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
        x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
        return [[x1, y1, x2, y2]]


    def detect_lanes(self, frame):
        edges = self.detect_edges(frame)
        #self.show_image("edges", edges)
        cropped_edges = self.region_of_interest(edges)
        #self.show_image("cropped edges", cropped_edges)
        line_segments = self.detect_line_segments(cropped_edges)
        print(line_segments)
        lanes = self.average_slope_intercept(frame, line_segments)

        return lanes


    def show_image(self, title, image):
        """
        Helper function to display images
        :param title: image title
        :param image: image
        """
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def display_lines(self, frame, lines, line_color=(0, 255, 0), line_width=20):
        """
        Return frame with lanes detected on top
        :param frame: current frame
        :param lines: lanes detected
        :param line_color: color to display
        :param line_width: width of lane detected
        :return: image with lanes on top
        """

        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
        line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        self.show_image("Frame with lines", line_image)
        return line_image


    # def process_video(self, cap):
    #     """
    #     :param cap: camera input given
    #     """
    #
    #     while True:
    #         # Read a frame from the camera
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #
    #         # Detect lanes in the current frame
    #         lanes = self.detect_lanes(frame)
    #
    #         # Display the frame with detected lanes
    #         frame_with_lanes = self.display_lines(frame, lanes)
    #
    #         # Show the processed frame
    #         cv2.imshow("Processed Video", frame_with_lanes)
    #
    #         # Break the loop when the 'q' key is pressed
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #
    #     # Release the camera capture object and close all windows
    #     cap.release()
    #     cv2.destroyAllWindows()









