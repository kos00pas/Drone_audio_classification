import threading
import numpy as np

class The_DOA_resp:
    def __init__(self, database, main_window):
        self.DATA = database
        self.main_window = main_window
        self.doa_thread = None
        self.stop_event = threading.Event()
        self.start_doa()
        self.now=False

    def for_destroy(self):
        if self.doa_thread is not None:
            self.now=True
            self.stop_event.set()  # Signal the thread to stop
            self.doa_thread.join(timeout=1)  # Wait for the thread to finish

    def start_doa(self):
        self.stop_event.clear()  # Clear the stop event in case it was previously set
        self.doa_thread = threading.Thread(target=self.doa_resp, daemon=True)
        self.doa_thread.start()

    def doa_resp(self):
        while not self.stop_event.is_set():
                self.main_window.axis_doa_resp.clear()
                doa_azimuth_degrees = self.DATA.resp4.doa_from_respeaker  # Replace with actual data retrieval method
                doa_azimuth_radians = np.radians(doa_azimuth_degrees)

                self.DATA.add_doa_to_log(doa_azimuth_degrees)

                self.main_window.axis_doa_resp.plot([0, doa_azimuth_radians], [0, 1], 'r-')
                self.main_window.axis_doa_resp.set_rmax(1)
                self.main_window.axis_doa_resp.set_yticklabels([])

                # Ensure ticks are properly set up
                self.main_window.axis_doa_resp.xaxis.set_ticks_position('none')
                self.main_window.axis_doa_resp.yaxis.set_ticks_position('none')

                self.main_window.canvas.draw_idle()
                self.main_window.doaa_label.config(text=f'DOA: {doa_azimuth_degrees}')
                # self.main_window.inner_frame_title.config(text=f"DOA: {doa_azimuth_degrees}", bg="white", font=("Arial", 15))
                if self.now:
                    break
                # Wait for the specified refresh interval
                self.stop_event.wait(self.DATA.doa_refresh)
        print("Exiting doa_resp loop")  # Debugging print


