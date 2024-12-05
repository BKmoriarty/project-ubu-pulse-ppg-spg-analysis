from datetime import datetime
import PySpin
import matplotlib.pyplot as plt
import keyboard

class AviType:
    """'Enum' to select AVI video type to be created and saved"""
    UNCOMPRESSED = 0
    MJPG = 1
    H264 = 2

class Record_video():
    def __init__(self,fps=30,exposure_time=6000,width = 200,height = 200,filename="VIDEO",time_record=20) -> None:
        self.fps = fps
        self.exposure_time = exposure_time
        self.width = width
        self.height = height
        self.chosenAviType = AviType.UNCOMPRESSED  # change me!
        self.filename = filename

        self.time_record = time_record
        self.frame_images = self.fps * self.time_record

        self.continue_recording = True
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
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

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
    
    def configure_exposure(self,cam):
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
            print('max exposure:',cam.ExposureTime.GetMax())
            exposure_time_to_set = self.exposure_time
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Shutter time set to %s us...\n' % exposure_time_to_set)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def configure_custom_image_settings(self, nodemap):
        """
        Configures a number of settings on the camera including offsets  X and Y, width,
        height, and pixel format. These settings must be applied before BeginAcquisition()
        is called; otherwise, they will be read only. Also, it is important to note that
        settings are applied immediately. This means if you plan to reduce the width and
        move the x offset accordingly, you need to apply such changes in the appropriate order.

        :param nodemap: GenICam nodemap.
        :type nodemap: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print('\n*** CONFIGURING CUSTOM IMAGE SETTINGS *** \n')

        try:
            result = True

            # Apply mono 8 pixel format
            #
            # *** NOTES ***
            # Enumeration nodes are slightly more complicated to set than other
            # nodes. This is because setting an enumeration node requires working
            # with two nodes instead of the usual one.
            #
            # As such, there are a number of steps to setting an enumeration node:
            # retrieve the enumeration node from the nodemap, retrieve the desired
            # entry node from the enumeration node, retrieve the integer value from
            # the entry node, and set the new value of the enumeration node with
            # the integer value from the entry node.
            #
            # Retrieve the enumeration node from the nodemap
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
            if PySpin.IsReadable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):

                # Retrieve the desired entry node from the enumeration node
                node_pixel_format_mono8 = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName('Mono8'))
                if PySpin.IsReadable(node_pixel_format_mono8):

                    # Retrieve the integer value from the entry node
                    pixel_format_mono8 = node_pixel_format_mono8.GetValue()

                    # Set integer as new value for enumeration node
                    node_pixel_format.SetIntValue(pixel_format_mono8)

                    print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())

                else:
                    print('Pixel format mono 8 not readable...')

            else:
                print('Pixel format not readable or writable...')
                
            # Set maximum width
            #
            # *** NOTES ***
            # Other nodes, such as those corresponding to image width and height,
            # might have an increment other than 1. In these cases, it can be
            # important to check that the desired value is a multiple of the
            # increment. However, as these values are being set to the maximum,
            # there is no reason to check against the increment.
            node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
            if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
                width_max = node_width.GetMax()

            else:
                print('Width not readable or writable...')

            # Set maximum height
            #
            # *** NOTES ***
            # A maximum is retrieved with the method GetMax(). A node's minimum and
            # maximum should always be a multiple of its increment.
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if  PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):

                height_max = node_height.GetMax()

            else:
                print('Height not readable or writable...')

            # Apply minimum to offset X
            #
            # *** NOTES ***
            # Numeric nodes have both a minimum and maximum. A minimum is retrieved
            # with the method GetMin(). Sometimes it can be important to check
            # minimums to ensure that your desired value is within range.
            node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX'))
            if PySpin.IsReadable(node_offset_x) and PySpin.IsWritable(node_offset_x):
                off_x_to_set = (width_max - self.width) // 2
                node_offset_x.SetValue(off_x_to_set)
                print('Offset X set to %i...' % off_x_to_set)

            else:
                print('Offset X not readable or writable...')

            # Apply minimum to offset Y
            #
            # *** NOTES ***
            # It is often desirable to check the increment as well. The increment
            # is a number of which a desired value must be a multiple of. Certain
            # nodes, such as those corresponding to offsets X and Y, have an
            # increment of 1, which basically means that any value within range
            # is appropriate. The increment is retrieved with the method GetInc().
            node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY'))
            if PySpin.IsReadable(node_offset_y) and PySpin.IsWritable(node_offset_y):
                
                off_y_to_set = (height_max - self.height) // 2
                node_offset_y.SetValue(off_y_to_set)
                print('Offset Y set to %i...' % off_y_to_set)

            else:
                print('Offset Y not readable or writable...')


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
            node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))

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

    def handle_close(self,evt):
        
        self.continue_recording = False

    def acquire_images(self,cam, nodemap):
        """
        This function acquires 30 images from a device, stores them in a list, and returns the list.
        please see the Acquisition example for more in-depth comments on acquiring images.

        :param cam: Camera to acquire images from.
        :param nodemap: Device nodemap.
        :type cam: CameraPtr
        :type nodemap: INodeMap
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        print('*** IMAGE ACQUISITION ***\n')
        try:
            result = True

            # Set acquisition mode to continuous
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False
            
            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
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
                print('\tUnable to set width; width for sequencer not available on all camera models...')

            # Set height; height recorded in pixels
            node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
            if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
                height_inc = node_height.GetInc()

                if self.height % height_inc != 0:
                    self.height = int(self.height / height_inc) * height_inc

                node_height.SetValue(self.height)

                print('\tHeight set to %d...' % node_height.GetValue())

            else:
                print('\tUnable to set height; height for sequencer not available on all camera models...')

            # Set exposure time; exposure time recorded in microseconds
            node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
            if not PySpin.IsReadable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
                print('Unable to set ExposureTime')
                return False

            exposure_time_max = node_exposure_time.GetMax()

            if self.exposure_time > exposure_time_max:
                self.exposure_time = exposure_time_max

            node_exposure_time.SetValue(self.exposure_time)

            print('\tExposure set to {0:.0f}...'.format(node_exposure_time.GetValue()))

            #  Begin acquiring images
            cam.BeginAcquisition()

            print('Acquiring images...')

            # Retrieve, convert, and save images
            images = list()

            # Create ImageProcessor instance for post processing images
            processor = PySpin.ImageProcessor()

            # Set default image processor color processing method
            #
            # *** NOTES ***
            # By default, if no specific color processing algorithm is set, the image
            # processor will default to NEAREST_NEIGHBOR method.
            processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)

            self.first_frame_time = datetime.timestamp(datetime.now())
            for i in range(self.frame_images):
                try:
                    # Time First Frame
                    
                    #  Retrieve next received image
                    image_result = cam.GetNextImage(1000)

                    #  Ensure image completion
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d...' % image_result.GetImageStatus())

                    else:
                        #  Print image information; height and width recorded in pixels
                        width = image_result.GetWidth()
                        height = image_result.GetHeight()
                        if i % 1000 == 0 :
                            print('Grabbed Image %d, width = %d, height = %d' % (i, width, height))

                        #  Convert image to mono 8 and append to list
                        images.append(processor.Convert(image_result, PySpin.PixelFormat_Mono8))

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

    def acquire_and_display_images(self,cam, nodemap, nodemap_tldevice):

        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
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
            print('\tUnable to set width; width for sequencer not available on all camera models...')

        # Set height; height recorded in pixels
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            height_inc = node_height.GetInc()

            if self.height % height_inc != 0:
                self.height = int(self.height / height_inc) * height_inc

            node_height.SetValue(self.height)

            print('\tHeight set to %d...' % node_height.GetValue())

        else:
            print('\tUnable to set height; height for sequencer not available on all camera models...')

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

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
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            if PySpin.IsReadable(node_device_serial_number):
                device_serial_number = node_device_serial_number.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

            # Close program
            print('Press enter to close the program..')

            # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
            fig = plt.figure(1)

            # Close the GUI when close event happens
            fig.canvas.mpl_connect('close_event', self.handle_close)

            # Retrieve and display images
            while(self.continue_recording):
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
                        print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

                    else:                    

                        # Getting the image data as a numpy array
                        image_data = image_result.GetNDArray()

                        # Draws an image on the current figure
                        plt.imshow(image_data, cmap='gray')

                        # Interval in plt.pause(interval) determines how fast the images are displayed in a GUI
                        # Interval is in seconds.
                        plt.pause(0.001)

                        # Clear current reference of a figure. This will improve display speed significantly
                        plt.clf()
                        
                        # If user presses enter, close the program
                        if keyboard.is_pressed('ENTER'):
                            print('Program is closing...')
                            
                            # Close figure
                            plt.close('all')             
                            # input('Done! Press Enter to exit...')
                            self.continue_recording=False                        

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
            node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

            if PySpin.IsReadable(node_serial):
                device_serial_number = node_serial.GetValue()
                print('Device serial number retrieved as %s...' % device_serial_number)

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
            print('Appending %d images to AVI file: %s.avi...' % (len(images), avi_filename))

            for i in range(len(images)):
                avi_recorder.Append(images[i])
                if i %1000 == 0 :
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

    def run_single_camera(self,cam):
        try:
            result = True

            # Retrieve TL device nodemap and print device information
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            result &= self.print_device_info(nodemap_tldevice)

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()
            
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

    def run_single_camera_show(self,cam):
        try:
            result = True

            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()

            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()
            
            # Configure exposure
            if not self.configure_exposure(cam):
                return False

            # Acquire images
            result &= self.acquire_and_display_images(cam, nodemap, nodemap_tldevice)

            # Deinitialize camera
            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def display_images(self):
        
        result = True

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

        for i, cam in enumerate(self.cam_list):

            print('Running example for camera %d...' % i)

            result &= self.run_single_camera(cam)
            print('Camera %d example complete... \n' % i)
        
        del cam

        # Clear camera list before releasing system
        self.cam_list.Clear()

        # Release instance
        self.system.ReleaseInstance()
        
        return result

if __name__ == '__main__':
    record = Record_video()
    record.init_camera()
    input("Tap show...")
    record.display_images()
    # record.start()