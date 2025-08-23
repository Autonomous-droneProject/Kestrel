import rclpy
from rclpy.node import Node
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from sensor_msgs.msg import Image
from kestrel_msgs.msg import TrackArray, Track as TrackMsg
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Set

from cv_bridge import CvBridge

def clip_bbox_to_image(bbox, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    """Clip [x1,y1,x2,y2] to image bounds; return None if invalid/empty."""
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def ns_to_time(timestamp_ns: int) -> Time:
    t = Time()
    t.sec = int(timestamp_ns // 1_000_000_000)
    t.nanosec = int(timestamp_ns % 1_000_000_000)
    return t

class InputNode(Node):
    """
    Subscribes to camera frames and TrackArray; time-matches them,
    shows per-ID crops in a GUI, and publishes a whitelist of selected IDs.
    """
    def __init__(self):
        super().__init__('input_node')
        
        # --- Buffers / state ---
        self.images: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.track_batches: "OrderedDict[int, List[TrackMsg]]" = OrderedDict()

        self.max_len: int = 30
        self.delta_ns: int = int(30e6)  # 30 ms tolerance
																			   
        self.current_ts: Optional[int] = None
        self.current_tracks_by_id: Dict[int, TrackMsg] = {}
        # display_items: id -> {"crop": np.ndarray, "conf": float, "class": str}
        self.display_items: Dict[int, Dict[str, object]] = {}

        # Selected (whitelisted) IDs
        self.selected_tracks: Set[int] = set()

        # GUI assets
        self.bridge = CvBridge()
        self.thumb_size = (128, 128)  # W,H for thumbnails
        self.tex_tags_by_id: Dict[int, str] = {}    # id -> texture tag
        self.image_item_by_id: Dict[int, str] = {}  # id -> image item tag
        self.gui_initialized = False

        # --- Initalize ros publishers/subscribers ---
        
        # Raw images from YOLO, with timestamp
        self.image_subscriber = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10) 
        # Array of Tracks from tracking_node -- each track is a bounding box with timestamp
        self.tracking_subscriber = self.create_subscription(TrackArray, '/kestrel/tracks', self.tracking_callback, 10)
        # Publish whitelisted array of tracks for the camera to track 
        self.pub_input = self.create_publisher(TrackArray, '/kestrel/whitelist_tracks', 10)

    # ----------------- Callbacks -----------------

    # receives raw image from /camera/image_raw
    def image_callback(self, msg: Image):
        # convert ROS stamp (sec, nanosec) to combined timestamp in nano sec
        ts = msg.header.stamp.sec * 1e9 + msg.header.nanosec
        
        # Convert to RGBA for dearpygui
        frame = self.bridge.imgmsg_to_cv2(msg, "rgba8")

        # Buffer and evict
        # mapping timestamp in nanosec as key, to the actual image as value
        self.images[ts] = frame
        while len(self.images) > self.max_len:
            self.images.popitem(last=False)

        # Try to match with an existing track batch
        self._try_match_after_new_image(ts)

    # receives TrackArray.msg from /kestrel/tracks
    def tracking_callback(self, msg: TrackArray):
        # All individual track timestamps should match TrackArray (msg).header timestamp
        ts = msg.header.stamp.sec * 1e9 + msg.header.nanosec
        
       # Buffer and evict
        self.track_batches[ts] = list(msg.tracks)
        while len(self.track_batches) > self.max_len:
            self.track_batches.popitem(last=False)

        # Try to match with an existing image
        self._try_match_after_new_tracks(ts)

    # ----------------- Matching & View Build -----------------
    def _find_nearest_key(self, target_ts: int, keys: List[int], tol_ns: int) -> Optional[int]:
        """Return key with smallest |key - target_ts| within tol_ns, else None."""
        if not keys:
            return None
        diffs = [(abs(k - target_ts), k) for k in keys]
        diffs.sort()
        best_diff, best_key = diffs[0]
        return best_key if best_diff <= tol_ns else None

    def _try_match_after_new_image(self, image_ts: int):
        # Find nearest track batch
        nearest_tracks_ts = self._find_nearest_key(image_ts, list(self.track_batches.keys()), self.delta_ns)
        if nearest_tracks_ts is None:
            return
        self._build_current_view(image_ts, nearest_tracks_ts)

    def _try_match_after_new_tracks(self, tracks_ts: int):
        # Find nearest image
        nearest_image_ts = self._find_nearest_key(tracks_ts, list(self.images.keys()), self.delta_ns)
        if nearest_image_ts is None:
            return
        self._build_current_view(nearest_image_ts, tracks_ts)

    def _build_current_view(self, image_ts: int, tracks_ts: int):
        """Set current view: compute crops for each track and refresh GUI."""
        frame = self.images.get(image_ts, None)
        tracks = self.track_batches.get(tracks_ts, None)
        if frame is None or tracks is None:
            return

        h, w = frame.shape[:2]
        self.current_ts = tracks_ts
        self.current_tracks_by_id = {}
        self.display_items = {}

        for t in tracks:
            # Map ID -> message
            self.current_tracks_by_id[int(t.id)] = t

            # Clip and crop
            clipped = clip_bbox_to_image((t.x1, t.y1, t.x2, t.y2), w, h)
            if clipped is None:
                continue  # skip invalid boxes
            x1, y1, x2, y2 = clipped
            crop = frame[y1:y2, x1:x2]

            # Resize to thumb for GUI
            if crop.size == 0:
                continue
            crop_resized = cv2.resize(crop, self.thumb_size, interpolation=cv2.INTER_LINEAR)

            self.display_items[int(t.id)] = {
                "crop": crop_resized,
                "conf": float(t.conf),
                "class": t.class_name
            }

        # Update GUI from display_items
        if self.gui_initialized:
            self._refresh_gui()
                    

    # ----------------- GUI -----------------

    def run_gui(self):
        """Build GUI once and then rely on _refresh_gui() to update textures."""
        dpg.create_context()
        dpg.create_viewport(title='Kestrel Tracking Management', width=900, height=700)

        with dpg.texture_registry(show=True):
            dpg.add_texture_registry(tag="tex_registry")

        with dpg.window(tag="Primary Window", label="Whitelist Tracks", width=880, height=660):
            dpg.set_primary_window("Primary Window", True)
            dpg.add_text("Click a thumbnail to toggle selection; selected items are highlighted.")
            dpg.add_separator()
            # A grid container; we'll add image buttons per ID
            dpg.add_child_window(tag="grid", width=-1, height=-1)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        self.gui_initialized = True

        # Initial fill (in case we had a match before GUI launched)
        self._refresh_gui()

        dpg.start_dearpygui()
        dpg.destroy_context()

        # On GUI close, publish current whitelist once (optional)
        self.publish_whitelist()

    def _refresh_gui(self):
        """Create/update textures & image widgets per track ID from self.display_items."""
        if not dpg.does_item_exist("grid"):
            return

        # Remove widgets for IDs no longer present
        for old_id in list(self.image_item_by_id.keys()):
            if old_id not in self.display_items:
                # delete image widget and texture
                if dpg.does_item_exist(self.image_item_by_id[old_id]):
                    dpg.delete_item(self.image_item_by_id[old_id])
                if dpg.does_item_exist(self.tex_tags_by_id[old_id]):
                    dpg.delete_item(self.tex_tags_by_id[old_id])
                self.image_item_by_id.pop(old_id, None)
                self.tex_tags_by_id.pop(old_id, None)

        # Add/update current IDs
        for tid, item in self.display_items.items():
            crop = item["crop"]  # RGBA 128x128
            # TODO: Ensure RGBA; if source is BGRA, convert here
            if crop.shape[2] == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)

            # Normalize to [0,1] and flatten for Dear PyGui texture
            tex_data = (crop.astype(np.float32) / 255.0).reshape(-1)

            # Create texture if missing
            if tid not in self.tex_tags_by_id or not dpg.does_item_exist(self.tex_tags_by_id[tid]):
                tex_tag = f"tex_{tid}"
                dpg.add_dynamic_texture(
                    width=self.thumb_size[0], height=self.thumb_size[1],
                    default_value=tex_data, tag=tex_tag, parent="tex_registry"
                )
                self.tex_tags_by_id[tid] = tex_tag
            else:
                dpg.set_value(self.tex_tags_by_id[tid], tex_data)

            # Create/Update image button widget
            selected = (tid in self.selected_tracks)
            tint = (0.2, 1.0, 0.2, 1.0) if selected else (1.0, 1.0, 1.0, 1.0)
            tooltip = f"ID {tid} | conf {item['conf']:.2f} | {item['class']}"

            if tid not in self.image_item_by_id or not dpg.does_item_exist(self.image_item_by_id[tid]):
                btn_tag = f"img_{tid}"
                with dpg.group(parent="grid"):
                    dpg.add_image_button(
                        texture_tag=self.tex_tags_by_id[tid],
                        tag=btn_tag,
                        width=self.thumb_size[0],
                        height=self.thumb_size[1],
                        tint_color=tint,
                        callback=self._on_tile_click,
                        user_data=tid
                    )
                    dpg.add_text(tooltip)
                self.image_item_by_id[tid] = btn_tag
            else:
                dpg.configure_item(self.image_item_by_id[tid], texture_tag=self.tex_tags_by_id[tid], tint_color=tint)
                # The text below the button isn't stored; simplest is to ignore or recreate group if needed.

    def _on_tile_click(self, sender, app_data, user_data):
        """Toggle selection for a given track ID and refresh the tile tint."""
        tid = int(user_data)
        if tid in self.selected_tracks:
            self.selected_tracks.remove(tid)
        else:
            self.selected_tracks.add(tid)
        # Update tint immediately
        if tid in self.image_item_by_id:
            tint = (0.2, 1.0, 0.2, 1.0) if tid in self.selected_tracks else (1.0, 1.0, 1.0, 1.0)
            dpg.configure_item(self.image_item_by_id[tid], tint_color=tint)

    # ----------------- Whitelist publisher -----------------

    def publish_whitelist(self):
        """Publish currently selected tracks as a TrackArray."""
        if not self.selected_tracks or not self.current_tracks_by_id or self.current_ts is None:
            return

        whitelisted = [self.current_tracks_by_id[tid] for tid in self.selected_tracks if tid in self.current_tracks_by_id]
        if not whitelisted:
            return

        arr = TrackArray()
        hdr = Header()
        hdr.stamp = ns_to_time(self.current_ts)
        # hdr.frame_id could be set if you carry it through your pipeline
        arr.header = hdr
        arr.tracks = whitelisted
        self.pub_input.publish(arr)

'''
def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
'''