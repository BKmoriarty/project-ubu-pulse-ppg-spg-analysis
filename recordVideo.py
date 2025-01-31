from datetime import datetime
import PySpin
import cv2
import keyboard
import threading
import queue
import numpy as np
from time import sleep

from SignalQualityAssessment import PPGQualityAssessment


class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2


class RecordVideo():
    def __init__(self, fps=120, exposure_time=2500, width=120, height=120, filename="VIDEO", time_record=20) -> None:
        self.fps = fps
        self.exposure_time = exposure_time
        self.max_exposure_time = 0
        self.width = width
        self.height = height
        self.chosenAviType = AviType.UNCOMPRESSED  # change me!
        self.filename = filename

        self.time_record = time_record
        self.frame_images = fps * time_record

        self.continue_recording = True
        self.image_queue = queue.Queue()
        self.image_list = []
        self.stop = False
        self.first_frame_time = None
        self.last_frame_time = None

        self.system = None
        self.cam_list = None

    def init_camera(self) -> bool:
        # Retrieve singleton reference to system object
        self.system = PySpin.System.GetInstance()

        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' %
              (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = self.system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected:', num_cameras)

        # Finish if there are no cameras
        if num_cameras == 0:
            # Clear camera list before releasing system
            cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

            print('Not enough cameras!')
            input('Done! Press Enter to exit...')
            return False

        self.cam_list = cam_list

        return True

    def configure_exposure(self, cam):
        """
        This function configures a custom exposure time. Automatic exposure is turned
        off in order to allow for the customization, and then the custom setting is
        applied.

        :param cam: Camera to configure exposure for.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """

        print('*** CONFIGURING EXPOSURE ***\n')

        try:
            result = True

            # Turn off automatic exposure mode
            #
            # *** NOTES ***
            # Automatic exposure prevents the manual configuration of exposure
            # times and needs to be turned off for this example. Enumerations
            # representing entry nodes have been added to QuickSpin. This allows
            # for the much easier setting of enumeration nodes to new values.
            #
            # The naming convention of QuickSpin enums is the name of the
            # enumeration node followed by an underscore and the symbolic of
            # the entry node. Selecting "Off" on the "ExposureAuto" node is
            # thus named "ExposureAuto_Off".
            #
            # *** LATER ***
            # Exposure time can be set automatically or manually as needed. This
            # example turns automatic exposure off to set it manually and back
            # on to return the camera to its default state.

            if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                print('Unable to disable automatic exposure. Aborting...')
                return False

            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            print('Automatic exposure disabled...')

            # Set exposure time manually; exposure time recorded in microseconds
            #
            # *** NOTES ***
            # Notice that the node is checked for availability and writability
            # prior to the setting of the node. In QuickSpin, availability and
            # writability are ensured by checking the access mode.
            #
            # Further, it is ensured that the desired exposure time does not exceed
            # the maximum. Exposure time is counted in microseconds - this can be
            # found out either by retrieving the unit with the GetUnit() method or
            # by checking SpinView.

            if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                print('Unable to set exposure time. Aborting...')
                return False

            # Ensure desired exposure time does not exceed the maximum
            print('max exposure:', cam.ExposureTime.GetMax())
            exposure_time_to_set = self.exposure_time
            exposure_time_to_set = min(
                cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Shutter time set to %s us...\n' % exposure_time_to_set)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def configure_custom_image_settings(self, cam):
        """
        Configures a number of settings on the camera including offsets X and Y,
        width, height, and pixel format. These settings must be applied before
        BeginAcquisition() is called; otherwise, those nodes would be read only.
        """
        print('\n*** CONFIGURING CUSTOM IMAGE SETTINGS ***\n')

        try:
            result = True

            # Set acquisition mode to continuous
            if cam.AcquisitionMode.GetAccessMode() == PySpin.RW:
                cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
                print('Acquisition mode set to continuous...')

            # Configure Frame Rate using QuickSpin API
            if cam.AcquisitionFrameRateEnable is not None:
                if cam.AcquisitionFrameRateEnable.GetAccessMode() == PySpin.RW:
                    # Enable frame rate control
                    cam.AcquisitionFrameRateEnable.SetValue(True)

                    # Set frame rate directly if available
                    if hasattr(cam, 'AcquisitionFrameRate') and cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW:
                        # Get limits
                        min_frame_rate = cam.AcquisitionFrameRate.GetMin()
                        max_frame_rate = cam.AcquisitionFrameRate.GetMax()
                        print(
                            f'Frame rate range: {min_frame_rate} to {max_frame_rate} fps')

                        # Set frame rate (make sure it's within range)
                        frame_rate = min(max_frame_rate, float(self.fps))
                        cam.AcquisitionFrameRate.SetValue(frame_rate)
                        print(f'Frame rate set to: {frame_rate} fps')

                        # Verify the frame rate was set
                        actual_frame_rate = cam.AcquisitionFrameRate.GetValue()
                        print(f'Actual frame rate: {actual_frame_rate} fps')
            else:
                print('Frame rate control not available for this camera model')

            # Apply mono 8 pixel format
            if cam.PixelFormat.GetAccessMode() == PySpin.RW:
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
                print('Pixel format set to %s...' %
                      cam.PixelFormat.GetCurrentEntry().GetSymbolic())
            else:
                print('Pixel format not available...')
                result = False

            # Get the camera's actual sensor dimensions
            nodemap = cam.GetNodeMap()
            sensor_width = PySpin.CIntegerPtr(nodemap.GetNode('SensorWidth'))
            sensor_height = PySpin.CIntegerPtr(nodemap.GetNode('SensorHeight'))

            if not PySpin.IsReadable(sensor_width) or not PySpin.IsReadable(sensor_height):
                print(
                    'Unable to read sensor dimensions. Using Width/Height max values instead.')
                max_width = cam.Width.GetMax()
                max_height = cam.Height.GetMax()
            else:
                max_width = sensor_width.GetValue()
                max_height = sensor_height.GetValue()
                print('Sensor dimensions: %dx%d' % (max_width, max_height))

            # Set width to width configured
            if cam.Width.GetAccessMode() == PySpin.RW and cam.Width.GetInc() != 0:
                # Calculate center offset for X
                center_offset_x = (max_width - self.width) // 2
                # Set width first
                cam.Width.SetValue(self.width)
                print('Width set to %i...' % cam.Width.GetValue())
                # Then set X offset to center
                if cam.OffsetX.GetAccessMode() == PySpin.RW:
                    cam.OffsetX.SetValue(center_offset_x)
                    print('Offset X set to %d...' % cam.OffsetX.GetValue())
            else:
                print('Width not available...')
                result = False

            # Set height to height configured
            if cam.Height.GetAccessMode() == PySpin.RW and cam.Height.GetInc() != 0:
                # Calculate center offset for Y
                center_offset_y = (max_height - self.height) // 2
                # Set height first
                cam.Height.SetValue(self.height)
                print('Height set to %i...' % cam.Height.GetValue())
                # Then set Y offset to center
                if cam.OffsetY.GetAccessMode() == PySpin.RW:
                    cam.OffsetY.SetValue(center_offset_y)
                    print('Offset Y set to %d...' % cam.OffsetY.GetValue())
            else:
                print('Height not available...')
                result = False

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def print_device_info(self, nodemap):
        """
        This function prints the device information of the camera from the transport
        layer; please see NodeMapInfo example for more in-depth comments on printing
        device information from the nodemap.

        :param nodemap: Transport layer device nodemap.
        :type nodemap: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print('\n*** DEVICE INFORMATION ***\n')

        try:
            result = True
            node_device_information = PySpin.CCategoryPtr(
                nodemap.GetNode('DeviceInformation'))

            if PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(),
                                      node_feature.ToString() if PySpin.IsReadable(node_feature) else 'Node not readable'))

            else:
                print('Device control information not readable.')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def handle_close(self, evt):

        self.continue_recording = False

    def acquire_images(self, cam, nodemap):

        print('*** IMAGE ACQUISITION ***\n')

        # Retrieve, convert, and save images
        images = list()

        try:
            result = True

            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(
                nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print(
                    'Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
                'Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print(
                    'Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            # Set width; width recorded in pixels
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
                width_inc = node_width.GetInc()

                if self.width % width_inc != 0:
                    self.width = int(self.width / width_inc) * width_inc

                node_width.SetValue(self.width)

                print('\tWidth set to {}...'.format(node_width.GetValue()))

            else:
                print(
                    '\tUnable to set width; width for sequencer not available on all camera models...')

            # Set height; height recorded in pixels
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
                height_inc = node_height.GetInc()

                if self.height % height_inc != 0:
                    self.height = int(self.height / height_inc) * height_inc

                node_height.SetValue(self.height)

                print('\tHeight set to %d...' % node_height.GetValue())

            else:
                print(
                    '\tUnable to set height; height for sequencer not available on all camera models...')

            ### Disable FrameRateAuto ###
            node_frame_rate_auto = PySpin.CEnumerationPtr(
                nodemap.GetNode("AcquisitionFrameRateAuto"))
            if not PySpin.IsAvailable(node_frame_rate_auto) or not PySpin.IsWritable(node_frame_rate_auto):
                print('Unable to turn off Frame Rate Auto (enum retrieval). Aborting...')
                return False

            node_frame_rate_auto_off = node_frame_rate_auto.GetEntryByName(
                "Off")
            if not PySpin.IsAvailable(node_frame_rate_auto_off) or not PySpin.IsReadable(node_frame_rate_auto_off):
                print(
                    'Unable to set Frame Rate Auto to Off (entry retrieval). Aborting...')
                return False

            frame_rate_auto_off = node_frame_rate_auto_off.GetValue()

            node_frame_rate_auto.SetIntValue(frame_rate_auto_off)

            print('Frame Rate Auto set to Off...')

            ### Enable AcquisitionFrameRateControlEnable ###
            node_acquisition_frame_rate_control_enable = PySpin.CBooleanPtr(
                nodemap.GetNode("AcquisitionFrameRateEnabled"))
            if not PySpin.IsAvailable(node_acquisition_frame_rate_control_enable) or not PySpin.IsWritable(node_acquisition_frame_rate_control_enable):
                print(
                    'Unable to turn on Acquisition Frame Rate Control Enable (bool retrieval). Aborting...')
                return False

            node_acquisition_frame_rate_control_enable.SetValue(True)

            print('Acquisiton Frame Rate Control Enabled...')

            ### Set AcquisitionFrameRate to 10 FPS ###
            if cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                print('Unable to set Frame Rate. Aborting...')
                return False

            cam.AcquisitionFrameRate.SetValue(self.fps)

            print('Acquisiton Frame Rate set to %s FPS...' % self.fps)

            # Set exposure time; exposure time recorded in microseconds
            node_exposure_time = PySpin.CFloatPtr(
                nodemap.GetNode('ExposureTime'))
            if not PySpin.IsReadable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
                print('Unable to set ExposureTime')
                return False

            exposure_time_max = node_exposure_time.GetMax()

            if self.exposure_time > exposure_time_max:
                self.exposure_time = exposure_time_max

            node_exposure_time.SetValue(self.exposure_time)

            print('\tExposure set to {0:.0f}...'.format(
                node_exposure_time.GetValue()))

            #  Begin acquiring images
            cam.BeginAcquisition()

            print('Acquiring images...')

            # Create ImageProcessor instance for post processing images
            processor = PySpin.ImageProcessor()

            # Set default image processor color processing method
            #
            # *** NOTES ***
            # By default, if no specific color processing algorithm is set, the image
            # processor will default to NEAREST_NEIGHBOR method.
            processor.SetColorProcessing(
                PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

            self.first_frame_time = datetime.timestamp(datetime.now())
            for i in range(self.frame_images):
                try:
                    # Time First Frame

                    #  Retrieve next received image
                    image_result = cam.GetNextImage(1000)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d...' %
                              image_result.GetImageStatus())

                    else:
                        self.image_queue.put(image_result)
                        #  Print image information; height and width recorded in pixels
                        width = image_result.GetWidth()
                        height = image_result.GetHeight()
                        if i % 100 == 0:
                            print('Grabbed Image %d/%d, width = %d, height = %d' %
                                  (i, self.frame_images, width, height))

                        #  Convert image to mono 8 and append to list
                        images.append(processor.Convert(
                            image_result, PySpin.PixelFormat_Mono8))

                        #  Release image
                        image_result.Release()
                        # print('')

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    result = False

            # End acquisition
            cam.EndAcquisition()
            self.last_frame_time = datetime.timestamp(datetime.now())
            self.stop = True

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result, images

    def acquire_and_display_images(self, cam, nodemap, nodemap_tldevice):

        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(
            sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

        # Set width; width recorded in pixels
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
            width_inc = node_width.GetInc()

            if self.width % width_inc != 0:
                self.width = int(self.width / width_inc) * width_inc

            node_width.SetValue(self.width)

            print('\tWidth set to {}...'.format(node_width.GetValue()))

        else:
            print(
                '\tUnable to set width; width for sequencer not available on all camera models...')

        # Set height; height recorded in pixels
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            height_inc = node_height.GetInc()

            if self.height % height_inc != 0:
                self.height = int(self.height / height_inc) * height_inc

            node_height.SetValue(self.height)

            print('\tHeight set to %d...' % node_height.GetValue())

        else:
            print(
                '\tUnable to set height; height for sequencer not available on all camera models...')

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(
                nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print(
                    'Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName(
                'Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print(
                    'Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            ### Disable FrameRateAuto ###
            node_frame_rate_auto = PySpin.CEnumerationPtr(
                nodemap.GetNode("AcquisitionFrameRateAuto"))
            if not PySpin.IsAvailable(node_frame_rate_auto) or not PySpin.IsWritable(node_frame_rate_auto):
                print('Unable to turn off Frame Rate Auto (enum retrieval). Aborting...')
                return False

            node_frame_rate_auto_off = node_frame_rate_auto.GetEntryByName(
                "Off")
            if not PySpin.IsAvailable(node_frame_rate_auto_off) or not PySpin.IsReadable(node_frame_rate_auto_off):
                print(
                    'Unable to set Frame Rate Auto to Off (entry retrieval). Aborting...')
                return False

            frame_rate_auto_off = node_frame_rate_auto_off.GetValue()

            node_frame_rate_auto.SetIntValue(frame_rate_auto_off)

            print('Frame Rate Auto set to Off...')

            ### Enable AcquisitionFrameRateControlEnable ###
            node_acquisition_frame_rate_control_enable = PySpin.CBooleanPtr(
                nodemap.GetNode("AcquisitionFrameRateEnabled"))
            if not PySpin.IsAvailable(node_acquisition_frame_rate_control_enable) or not PySpin.IsWritable(node_acquisition_frame_rate_control_enable):
                print(
                    'Unable to turn on Acquisition Frame Rate Control Enable (bool retrieval). Aborting...')
                return False

            node_acquisition_frame_rate_control_enable.SetValue(True)

            print('Acquisiton Frame Rate Control Enabled...')

            ### Set AcquisitionFrameRate to 10 FPS ###
            if cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                print('Unable to set Frame Rate. Aborting...')
                return False

            cam.AcquisitionFrameRate.SetValue(self.fps)

            print('Acquisiton Frame Rate set to %s FPS...' % self.fps)

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.
            cam.BeginAcquisition()

            print('Acquiring images...')

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(
                nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' %
                      device_serial_number)

            # Close program
            print('Press enter to close the program..')

            # Retrieve and display images
            while (self.continue_recording):
                try:

                    #  Retrieve next received image
                    #
                    #  *** NOTES ***
                    #  Capturing an image houses images on the camera buffer. Trying
                    #  to capture an image that does not exist will hang the camera.
                    #
                    #  *** LATER ***
                    #  Once an image from the buffer is saved and/or no longer
                    #  needed, the image must be released in order to keep the
                    #  buffer from filling up.

                    image_result = cam.GetNextImage(1000)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d ...' %
                              image_result.GetImageStatus())

                    else:
                        # Getting the image data as a numpy array
                        image_data = image_result.GetNDArray()

                        self.image_queue.put(image_data)
                        # self.image_list.append(image_data)

                        # Display the image using OpenCV
                        cv2.imshow('Camera Feed', image_data)

                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                        # Break the loop if 'Enter' is pressed
                        if keyboard.is_pressed('enter'):
                            print('Program is closing...')

                            # Close figure
                            cv2.destroyAllWindows()
                            # input('Done! Press Enter to exit...')
                            self.continue_recording = False

                    #  Release image
                    #
                    #  *** NOTES ***
                    #  Images retrieved directly from the camera (i.e. non-converted
                    #  images) need to be released in order to keep from filling the
                    #  buffer.
                    image_result.Release()

                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False

            # Clean up
            cv2.destroyAllWindows()
            #  End acquisition
            #
            #  *** NOTES ***
            #  Ending acquisition appropriately helps ensure that devices clean up
            #  properly and do not need to be power-cycled to maintain integrity.
            cam.EndAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return True

    def calculate_and_plot_mean(self):
        import matplotlib.pyplot as plt
        import numpy as np
        import time

        # Initialize interactive plotting
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        frames_buffer = []
        last_plot_time = time.time()

        # Set up plot titles and labels
        ax1.set_title('Time Domain Signal')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Amplitude')

        while self.continue_recording:
            if not self.image_queue.empty():
                # Retrieve image data from the queue
                image_data = self.image_queue.get()
                # Calculate mean value of the image data
                mean_value = np.mean(image_data)
                frames_buffer.append(mean_value)

                current_time = time.time()
                # Plot every 5 seconds
                if current_time - last_plot_time >= 5:
                    if frames_buffer:
                        # Clear previous plots and plot time domain signal
                        ax1.clear()
                        ax1.plot(frames_buffer, color='blue',
                                 label='Mean Amplitude')
                        ax1.set_title('Time Domain Signal')
                        ax1.set_xlabel('Frame')
                        ax1.set_ylabel('Amplitude')
                        ax1.legend()

                        # Plot frequency spectrum
                        self.plot_spectrum(frames_buffer, ax2)

                        # Update plots
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()

                        # Clear buffer for next interval
                        frames_buffer.clear()
                        last_plot_time = current_time

            # Print the current size of the image queue for monitoring
            # print("queue size:", self.image_queue.qsize(),'exposure time:',self.exposure_time)
            print('exposure time: %d/%d' %
                  (self.exposure_time, self.max_exposure_time))
            plt.pause(0.001)  # Small pause to allow GUI updates

    def plot_spectrum(self, signal, ax):
        """
        Plot the frequency spectrum of the given signal.

        :param signal: The input time-domain signal
        :param ax: The matplotlib axis to plot on
        """
        import numpy as np

        if len(signal) < 2:
            print("Signal is too short to compute spectrum.")
            return

        # Compute the Fast Fourier Transform (FFT) of the signal
        fps = self.fps  # Use the actual sampling rate
        n = len(signal)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, d=1/fps)

        # Filter to only positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitude = np.abs(fft_result)[pos_mask]

        # Clear any previous plot on the axis
        ax.clear()

        # Plot the frequency spectrum
        ax.plot(freqs, magnitude, label='Magnitude Spectrum',
                color='b', linewidth=1.5)
        ax.set_title('Frequency Spectrum', fontsize=14)
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.axvline(x=0.8, color='r', linestyle='--',
                   label='Lower Bound (0.8 Hz)')
        ax.axvline(x=4, color='g', linestyle='--', label='Upper Bound (4 Hz)')

        # Set x-axis limits to a maximum of 10 Hz
        ax.set_xlim(0, min(10, max(freqs)))

        # Set y-axis limits with a focus on typical PPG frequency ranges
        ax.set_ylim(0, 600)

        # Enable legend
        ax.legend(loc='upper right', fontsize=10)

    def save_list_to_avi(self, nodemap, nodemap_tldevice, images):
        """
        This function prepares, saves, and cleans up an AVI video from a vector of images.

        :param nodemap: Device nodemap.
        :param nodemap_tldevice: Transport layer device nodemap.
        :param images: List of images to save to an AVI video.
        :type nodemap: INodeMap
        :type nodemap_tldevice: INodeMap
        :type images: list of ImagePtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print('*** CREATING VIDEO ***')

        try:
            result = True

            # Retrieve device serial number for filename
            device_serial_number = ''
            node_serial = PySpin.CStringPtr(
                nodemap_tldevice.GetNode('DeviceSerialNumber'))

            if PySpin.IsReadable(node_serial):
                device_serial_number = node_serial.GetValue()
                print('Device serial number retrieved as %s...' %
                      device_serial_number)

            # Get the current frame rate; acquisition frame rate recorded in hertz
            #
            # *** NOTES ***
            # The video frame rate can be set to anything; however, in order to
            # have videos play in real-time, the acquisition frame rate can be
            # retrieved from the camera.

            # node_acquisition_framerate = PySpin.CFloatPtr(nodemap.GetNode('AcquisitionFrameRate'))

            # if not PySpin.IsReadable(node_acquisition_framerate):
            #     print('Unable to retrieve frame rate. Aborting...')
            #     return False

            # framerate_to_set = node_acquisition_framerate.GetValue()

            framerate_to_set = self.fps

            print('Frame rate to be set to %d...' % framerate_to_set)

            # Select option and open AVI filetype with unique filename
            #
            # *** NOTES ***
            # Depending on the filetype, a number of settings need to be set in
            # an object called an option. An uncompressed option only needs to
            # have the video frame rate set whereas videos with MJPG or H264
            # compressions should have more values set.
            #
            # Once the desired option object is configured, open the AVI file
            # with the option in order to create the image file.
            #
            # Note that the filename does not need to be appended to the
            # name of the file. This is because the AVI recorder object takes care
            # of the file extension automatically.
            #
            # *** LATER ***
            # Once all images have been added, it is important to close the file -
            # this is similar to many other standard file streams.

            avi_recorder = PySpin.SpinVideo()

            if self.chosenAviType == AviType.UNCOMPRESSED:
                avi_filename = 'SaveToAvi-Uncompressed-%s' % device_serial_number

                option = PySpin.AVIOption()
                option.frameRate = framerate_to_set
                option.height = images[0].GetHeight()
                option.width = images[0].GetWidth()

            elif self.chosenAviType == AviType.MJPG:
                avi_filename = 'SaveToAvi-MJPG-%s' % device_serial_number

                option = PySpin.MJPGOption()
                option.frameRate = framerate_to_set
                option.quality = 75
                option.height = images[0].GetHeight()
                option.width = images[0].GetWidth()

            elif self.chosenAviType == AviType.H264:
                avi_filename = 'SaveToAvi-H264-%s' % device_serial_number

                option = PySpin.H264Option()
                option.frameRate = framerate_to_set
                option.bitrate = 1000000
                option.height = images[0].GetHeight()
                option.width = images[0].GetWidth()

            else:
                print('Error: Unknown AviType. Aborting...')
                return False

            avi_filename = self.filename

            avi_recorder.Open(avi_filename, option)

            # Construct and save AVI video
            #
            # *** NOTES ***
            # Although the video file has been opened, images must be individually
            # appended in order to construct the video.
            print('Appending %d images to AVI file: %s.avi...' %
                  (len(images), avi_filename))

            for i in range(len(images)):
                avi_recorder.Append(images[i])
                if i % 1000 == 0:
                    print('Appended image %d...' % i)

            # Close AVI file
            #
            # *** NOTES ***
            # Once all images have been appended, it is important to close the
            # AVI file. Notice that once an AVI file has been closed, no more
            # images can be added.

            avi_recorder.Close()
            print('Video saved at %s.avi' % avi_filename)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

        return result

    def run_single_camera(self, cam):
        try:
            result = True

            # Retrieve TL device nodemap and print device information
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            result &= self.print_device_info(nodemap_tldevice)

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            # Configure image settings
            if not self.configure_custom_image_settings(cam):
                return False

            # Configure exposure
            if not self.configure_exposure(cam):
                return False

            # Acquire list of images
            err, images = self.acquire_images(cam, nodemap)
            if err < 0:
                return err

            result &= self.save_list_to_avi(nodemap, nodemap_tldevice, images)

            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def start_acquisition(self, cam):
        """
        Starts the acquisition thread and listens for keyboard inputs for adjustments.
        """

        while self.continue_recording:
            self.max_exposure_time = cam.ExposureTime.GetMax()
            for i in range(1, 10):
                if keyboard.is_pressed(str(i)):
                    self.exposure_time = int(
                        cam.ExposureTime.GetMax() * i / 10)
                    print(f'Adjusting exposure to: {self.exposure_time} us')
                    self.configure_exposure(cam)

            if keyboard.is_pressed('0'):
                self.exposure_time = cam.ExposureTime.GetMax()
                print(
                    f'Adjusting exposure to maximum: {self.exposure_time} us')
                self.configure_exposure(cam)

            if keyboard.is_pressed('q'):
                self.continue_recording
                break

            sleep(0.01)

    def run_single_camera_show(self, cam):
        try:
            result = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            # Configure image settings
            if not self.configure_custom_image_settings(cam):
                return False

            # Configure exposure
            if not self.configure_exposure(cam):
                return False

            # Acquire images
            # result &= self.acquire_and_display_images(cam, nodemap, nodemap_tldevice)

            acquisition_thread = threading.Thread(
                target=self.acquire_and_display_images, args=(cam, nodemap, nodemap_tldevice))
            exposure_thread = threading.Thread(
                target=self.start_acquisition, args=(cam,))

            acquisition_thread.start()
            exposure_thread.start()

            self.calculate_and_plot_mean()  # Start the plotting on the main thread

            acquisition_thread.join()
            exposure_thread.join()

            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def display_images(self):

        result = True

        if self.cam_list == None:
            print('Not enough cameras!')
            return False

        for i, cam in enumerate(self.cam_list):

            print('Running example for camera %d...' % i)

            result &= self.run_single_camera_show(cam)
            print('Camera %d example complete... \n' % i)

        # del cam

        # # Clear camera list before releasing system
        # self.cam_list.Clear()

        # # Release instance
        # self.system.ReleaseInstance()

        return result

    def start(self) -> bool:

        result = True

        if self.cam_list == None:
            print('Not enough cameras!')
            return False

        for i, cam in enumerate(self.cam_list):

            print('Running example for camera %d...' % i)

            result &= self.run_single_camera(cam)
            print('Camera %d example complete... \n' % i)

        # del cam

        # # Clear camera list before releasing system
        # self.cam_list.Clear()

        # # Release instance
        # self.system.ReleaseInstance()

        return result


if __name__ == '__main__':
    record = RecordVideo()
    record.init_camera()
    # input("Tap show...")
    record.display_images()
    # record.start()
