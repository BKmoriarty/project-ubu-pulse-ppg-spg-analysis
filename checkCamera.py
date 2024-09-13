
from matplotlib import pyplot as plt
from simple_pyspin import Camera
from PIL import Image
import cv2
import serial
import threading
import pandas as pd
import os
import time
import shutil
import numpy as np
from datetime import datetime

from SignalExtractionAndAnalysis import Analysis_PPG_SPG


class Main():
    def __init__(self, exposure_time=2200):
        # self.cap = EasyPySpin.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, 1500)

        self.frame = 600
        self.fps = 30
        self.ExposureTime = exposure_time
        self.stop = False

        self.video_path = 'video.avi'
        self.serial_path = 'serial.xlsx'
        self.first_frame_time = None
        self.last_frame_time = None

        self.custom_header = ['Data', 'Time', 'Value',
                              'Status', 'Checksum', "Hex Data", "Raw Data"]

        self.port_name = 'COM3'
        self.baudrate = 112500
        self.ser = serial.Serial(port=self.port_name, baudrate=self.baudrate, parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE, bytesize=serial.EIGHTBITS, timeout=1)

        # check anything is connected to the port
        if not self.ser.is_open:
            print("Serial can't open\nexit")
            return -1

        self.image_folder = 'frames'
        try:
            if os.path.isfile(self.video_path) or os.path.islink(self.video_path):
                os.unlink(self.video_path)
            elif os.path.isdir(self.video_path):
                shutil.rmtree(self.video_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (self.video_path, e))
        try:
            if os.path.isfile(self.serial_path) or os.path.islink(self.serial_path):
                os.unlink(self.serial_path)
            elif os.path.isdir(self.serial_path):
                shutil.rmtree(self.serial_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (self.serial_path, e))
        os.makedirs(self.image_folder, exist_ok=True)
        for filename in os.listdir(self.image_folder):
            file_path = os.path.join(self.image_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def __del__(self):
        self.ser.close()
        cv2.destroyAllWindows()

    # def save_video(self):
    #     start_time = time.time()
    #     # self.current_frame <= self.frame:
    #     all_frame = []
    #     while (time.time() - start_time) <= 20:
    #         ret, frame = self.cap.read()
    #         if not ret:
    #             print("Failed to grab frame")
    #             break

    #         frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)

    #         all_frame.append(frame)
    #         print(time.time() - start_time)

    #         # img_show = cv2.resize(frame, None, fx=0.25, fy=0.25)
    #         # cv2.imshow("press q to quit", img_show)

    #         self.current_frame += 1

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord("q"):
    #             break

    #     print(f"png cached : {(time.time() - start_time):.6f}")
    #     # Save frame as PNG
    #     # for frame in all_frame:
    #     #     cv2.imwrite(os.path.join(self.image_folder,
    #     #                              f'frame_{frame[0]:04d}.png'), frame[1])
    #     self.stop = True
    #     cv2.destroyAllWindows()
    #     print("Png saved")

    #     frame = all_frame[0]
    #     height, width, layers = frame.shape
    #     video = cv2.VideoWriter(
    #         self.video_path, cv2.VideoWriter_fourcc(*'XVID'), 42, (width, height))
    #     x = 0
    #     for image in all_frame:
    #         x += 1
    #         print(f"image: {x}")
    #         video.write(image)

    #     video.release()
    #     print("Video saved")

    def save_video_by_pyspin(self):
        time.sleep(1)
        with Camera() as cam:
            cam.Width = self.frame
            cam.Height = self.frame

            cam.AcquisitionFrameRateAuto = 'Off'
            cam.AcquisitionFrameRateEnabled = True
            cam.AcquisitionFrameRate = self.fps

            cam.ExposureAuto = 'Off'
            cam.ExposureTime = self.ExposureTime

            start_time = time.time()
            print(f"record camera :{pd.Timestamp.now().time()}")
            cam.start()
            # imgs = [cam.get_array() for n in range(600)]
            # List to store images and their timestamps
            imgs_with_timestamps = []
            temp = []

            for n in range(600):
                img_array = cam.get_array()
                temp.append(img_array)
                timestamp = datetime.timestamp(datetime.now())
                imgs_with_timestamps.append((timestamp, img_array))

                if n == 0:
                    self.first_frame_time = timestamp
                self.last_frame_time = timestamp

            cam.stop()
            end_time = time.time()

            np.save('temp\img_array.npy', img_array)
            print(f'Time Record : {(end_time-start_time):.6f} Second.')

        time.sleep(1)
        self.stop = True

        # for n, img in enumerate(imgs):
        #     Image.fromarray(img).save(os.path.join(
        #         self.image_folder, '%08d.png' % n))

        for timestamp, img_array in imgs_with_timestamps:
            filename = os.path.join(self.image_folder, '%s.png' % timestamp)
            Image.fromarray(img_array).save(filename)

    def convert_frames_to_video(self):
        images = [img for img in os.listdir(
            self.image_folder) if img.endswith(".png")]
        images.sort()

        frame = cv2.imread(os.path.join(self.image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(
            self.video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (width, height))

        for image in images:
            # print(f"image: {x}")
            # img = cv2.imread(os.path.join(self.image_folder, image))
            # timestamp = pd.Timestamp.now().time()
            # cv2.putText(img, f"Time: {timestamp}", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # video.write(img)
            # Extract timestamp from filename
            timestamp_str = os.path.splitext(image)[0]
            try:
                # Convert timestamp_str to a readable format
                timestamp = datetime.fromtimestamp(
                    float(timestamp_str)).strftime('%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                print(timestamp_str)
                timestamp = "Invalid Timestamp"

            img = cv2.imread(os.path.join(self.image_folder, image))
            # Overlay the timestamp on the image
            cv2.putText(img, f"Time: {timestamp}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            video.write(img)

        video.release()
        print("Video saved")

    def save_serial(self):
        serial_data = []
        start_time = time.time()
        print(f"record serial {pd.Timestamp.now().time()}")
        while (not self.stop):
            if self.ser.in_waiting > 0 and not self.stop:
                data = self.ser.read(100)  # Read 100 bytes or adjust as needed
                packets = [data[i:i+4]
                           for i in range(0, len(data), 4)]
                for packet in packets:
                    if packet[0] == 0xFF and not self.stop:
                        value = packet[1]
                        status = packet[2]
                        checksum = packet[3]
                        hex_data = packet.hex()
                        raw_data = data
                        # print(
                        #     f"value: {value}, status: {status}, checksum: {checksum}, hex_data: {hex_data}, raw_data: {raw_data}")
                        serial_data.append(
                            [pd.Timestamp.now().time(), datetime.timestamp(datetime.now()), value, status, checksum, hex_data, raw_data])

        print(f'Time Excel : {(time.time()-start_time):.6f} Second.')
        if len(serial_data) > 0:
            df = pd.DataFrame(serial_data, columns=self.custom_header)

            # Time type to float
            df['Time'] = df['Time'].astype(float)

            print(f"first_frame_time: {self.first_frame_time}")
            print(f'first_row_time: {df.iloc[0]["Time"]}')
            print(f"last_frame_time: {self.last_frame_time}")
            print(f'last_row_time: {df.iloc[-1]["Time"]}')

            start_date = float(self.first_frame_time)
            end_date = float(self.last_frame_time)

            filtered_df = df[(df['Time'] >= start_date)
                             & (df['Time'] <= end_date)]

            print(f'lenght: {len(df)} -> {len(filtered_df)}')
            time_old = df.iloc[-1]["Time"]-df.iloc[0]['Time']
            time_new = filtered_df.iloc[-1]["Time"]-filtered_df.iloc[0]['Time']
            print(f"Total time: {time_old} -> {time_new}")

            # Use a context manager to write the Excel file
            with pd.ExcelWriter(self.serial_path, engine='openpyxl') as writer:
                filtered_df.to_excel(
                    writer, index_label='ID', sheet_name='Sheet1')

        print("Serial data saved")

    def show_real_time_video(self):
        with Camera() as cam:
            cam.Width = self.frame
            cam.Height = self.frame

            cam.AcquisitionFrameRateAuto = 'Off'
            cam.AcquisitionFrameRateEnabled = True
            cam.AcquisitionFrameRate = self.fps

            cam.ExposureAuto = 'Off'
            cam.ExposureTime = self.ExposureTime

            cam.start()

            while True:
                img_array = cam.get_array()

                cv2.imshow('Real Time Video', img_array)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cam.stop()
            cv2.destroyAllWindows()

    def run(self):
        self.show_real_time_video()

        thread1 = threading.Thread(
            target=self.save_video_by_pyspin, name='video')
        thread2 = threading.Thread(target=self.save_serial, name='serial')

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        if self.stop:
            self.convert_frames_to_video()


if __name__ == '__main__':
    exposure_time = 2000

    main = Main(exposure_time)
    main.run()

    analysis = Analysis_PPG_SPG(
        "./video.avi", "./serial.xlsx", 150, exposure_time)
    a, b, c = analysis.main()
    plt.show()