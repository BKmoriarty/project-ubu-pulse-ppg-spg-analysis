
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
from recordVideo import RecordVideo


class Main():
    def __init__(self, port_name='COM10', exposure_time=2200, name: str = 'unknown', fps=30, size=(600, 600), lenght=20):
        
        self.frame: tuple = size
        self.fps = fps
        self.length = lenght
        self.ExposureTime = exposure_time

        self.current_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.video_path = 'storage/%s %s %s %s %s/video' % (
            self.current_time, name, exposure_time, fps, size)
        self.serial_path = 'storage/%s %s %s %s %s/serial.xlsx' % (
            self.current_time, name, exposure_time, fps, size)

        self.custom_header = ['Data', 'Time', 'Value',
                              'Status', 'Checksum', "Hex Data", "Raw Data"]

        # Initialize RecordVideo instance
        self.camera = RecordVideo(
            fps=fps,
            exposure_time=exposure_time,
            width=size[0],
            height=size[1],
            filename=self.video_path,
            time_record=lenght
        )
        self.camera.init_camera()

        # Serial port setup
        self.port_name = port_name
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
        os.makedirs('storage', exist_ok=True)
        os.makedirs('temp', exist_ok=True)

    def __del__(self):
        self.ser.close()
        cv2.destroyAllWindows()

    def save_serial(self):
        serial_data = []
        start_time = time.time()
        print(f"record serial {pd.Timestamp.now().time()}")
        while (not self.camera.stop):
            if self.ser.in_waiting > 0 and not self.camera.stop:
                data = self.ser.read(100)  # Read 100 bytes or adjust as needed
                packets = [data[i:i+4]
                           for i in range(0, len(data), 4)]
                for packet in packets:
                    if packet[0] == 0xFF and not self.camera.stop:
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
            df = pd.DataFrame(serial_data, columns=pd.Index(self.custom_header))

            # Time type to float
            df['Time'] = df['Time'].astype(float)

            print(f"first_frame_time: {self.camera.first_frame_time}")
            print(f'first_row_time: {df.iloc[0]["Time"]}')
            print(f"last_frame_time: {self.camera.last_frame_time}")
            print(f'last_row_time: {df.iloc[-1]["Time"]}')

            if self.camera.first_frame_time == None or self.camera.last_frame_time == None:
                print("Error: first_frame_time or last_frame_time is None")
                return

            start_date = float(self.camera.first_frame_time)
            end_date = float(self.camera.last_frame_time)

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

    
        
    def run(self):
        self.camera.display_images()

        os.makedirs(
            f'storage/{self.current_time} {name} {exposure_time} {fps} {self.frame}', exist_ok=True)

        thread1 = threading.Thread(target=self.camera.start, name='camera')
        thread2 = threading.Thread(target=self.save_serial, name='serial')

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        return self.video_path+"-0000.avi", self.serial_path


def calculate_exposure_time(new_fps: int = 88):
    return 6000 * (88/new_fps)


if __name__ == '__main__':
    """
        w x h   -> fps
        > 500   -> 30 FPS
        500x500 -> 48 FPS
        400x400 -> 58 FPS
        300x300 -> 70 FPS
        200x200 -> 88 FPS
        148x148 -> 100 FPS
        100x100 -> 122 FPS
        50x50   -> 148 FPS

        best: 200x200 -> 88 FPS
                size_ppg = 200  # W x H
                size_spg = 100  # W x H
                exposure_time = 6000  # us
    """

    port = "COM4"
    import serial.tools.list_ports

    def get_available_ports():
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    available_ports = get_available_ports()
    print("Available ports:", available_ports)

    if available_ports:
        port = available_ports[0]  # Select the first available port
    else:
        raise Exception("No available ports found")

    # * 400x400 -> 58 FPS
    # exposure_time = 6000  # us
    # size_ppg = 200  # W x H
    # size_spg = 100  # W x H
    # fps = 58  # Hz
    # cache = False

    # * 200x200 -> 88 FPS
    # exposure_time = 6000  # us
    # size_ppg = 200  # W x H
    # size_spg = 100  # W x H
    # fps = 88  # Hz
    # cache = False

    # * 148x148 -> 100 FPS
    # exposure_time = 6000  # us
    # size_ppg = 148  # W x H
    # size_spg = 100  # W x H
    # fps = 100  # Hz

    # * CM3-U3-31S4M-CS 200x200 -> 350 FPS
    size_cam = (200, 200)  # W x H
    size_ppg = 200  # W x H
    size_spg = 100  # W x H
    fps = 300  # Hz
    exposure_time = 1500 # 2500
    # int(calculate_exposure_time(fps))
    cache = False

    length = 20  # second

    print(f"Exposure Time: {exposure_time} us")

    name = input("Enter your name: ")
    main = Main(port, exposure_time, name, fps, size_cam, length)
    video_path, serial_path = main.run()

    print(f'path: "{video_path}", "{serial_path}"')

    analysis = Analysis_PPG_SPG(
        video_path, serial_path, size_ppg, size_spg, exposure_time, fps)
    ppg, spg, excel = analysis.main()
    plt.show()

    # input feedback
    feedback_rate = input("Enter feedback rate 1-10: ")
    feedback = input("Enter feedback: ")

    # rename folder
    folder_name = f"storage/{main.current_time} {name} {exposure_time} {fps} {size_cam} rate-{feedback_rate} {feedback}"
    os.rename(
        f"storage/{main.current_time} {name} {exposure_time} {fps} {size_cam}", folder_name)
