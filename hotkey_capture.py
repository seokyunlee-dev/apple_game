"""
Hotkey + drag-to-select screen region capture (Windows).

Usage:
  python hotkey_capture.py

Hotkey:
  Ctrl + Alt + S  (then drag)
  Ctrl + Alt + Q  (quit)

Notes:
  - Requires: Pillow, pynput
  - Saves PNGs into ./captures
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Optional, Tuple


def _requirements_hint() -> str:
    return "pip install pillow pynput"


try:
    from PIL import ImageGrab  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Pillow가 필요합니다. 다음을 실행하세요:\n"
        f"  {_requirements_hint()}\n\n"
        f"원인: {e}"
    )

try:
    from pynput import keyboard  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "pynput이 필요합니다. 다음을 실행하세요:\n"
        f"  {_requirements_hint()}\n\n"
        f"원인: {e}"
    )

import ctypes
import tkinter as tk


@dataclass(frozen=True)
class VirtualScreen:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def get_virtual_screen() -> VirtualScreen:
    """
    Windows 가상 스크린(멀티모니터 전체) 좌표/크기 반환.
    """
    user32 = ctypes.windll.user32
    # SM_XVIRTUALSCREEN=76, SM_YVIRTUALSCREEN=77, SM_CXVIRTUALSCREEN=78, SM_CYVIRTUALSCREEN=79
    left = int(user32.GetSystemMetrics(76))
    top = int(user32.GetSystemMetrics(77))
    width = int(user32.GetSystemMetrics(78))
    height = int(user32.GetSystemMetrics(79))
    return VirtualScreen(left=left, top=top, width=width, height=height)


@dataclass
class CaptureResult:
    bbox: Tuple[int, int, int, int]  # (left, top, right, bottom) in screen coords
    image_path: str
    meta_path: str


class RegionSelector:
    def __init__(self, root: tk.Tk, out_dir: Path) -> None:
        self.root = root
        self.out_dir = out_dir
        self.vs = get_virtual_screen()

        self._overlay: Optional[tk.Toplevel] = None
        self._canvas: Optional[tk.Canvas] = None
        self._rect_id: Optional[int] = None
        self._start_xy: Optional[Tuple[int, int]] = None

        self._active = False
        self._last_result: Optional[CaptureResult] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_result(self) -> Optional[CaptureResult]:
        return self._last_result

    def consume_last_result(self) -> Optional[CaptureResult]:
        """
        가장 최근 캡처 결과를 1회성으로 가져오고 내부 상태를 비웁니다.
        """
        res = self._last_result
        self._last_result = None
        return res

    def cancel(self) -> None:
        self._cancel()

    def begin(self) -> None:
        if self._active:
            return
        self._active = True
        self._last_result = None

        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.attributes("-topmost", True)
        overlay.attributes("-alpha", 0.25)
        overlay.configure(bg="black")

        overlay.geometry(f"{self.vs.width}x{self.vs.height}+{self.vs.left}+{self.vs.top}")
        overlay.focus_force()
        overlay.bind("<Escape>", lambda _e: self._cancel())

        canvas = tk.Canvas(overlay, bg="black", highlightthickness=0, cursor="crosshair")
        canvas.pack(fill="both", expand=True)
        canvas.bind("<ButtonPress-1>", self._on_down)
        canvas.bind("<B1-Motion>", self._on_move)
        canvas.bind("<ButtonRelease-1>", self._on_up)

        self._overlay = overlay
        self._canvas = canvas
        self._rect_id = None
        self._start_xy = None

    def _cancel(self) -> None:
        self._destroy_overlay()
        self._active = False

    def _destroy_overlay(self) -> None:
        try:
            if self._overlay is not None:
                self._overlay.destroy()
        finally:
            self._overlay = None
            self._canvas = None
            self._rect_id = None
            self._start_xy = None

    def _on_down(self, event: tk.Event) -> None:
        # event.x/event.y 는 overlay(가상 화면 origin 기준) 상대 좌표
        x0 = int(event.x)
        y0 = int(event.y)
        self._start_xy = (x0, y0)

        if self._canvas is None:
            return
        if self._rect_id is not None:
            self._canvas.delete(self._rect_id)
        self._rect_id = self._canvas.create_rectangle(
            x0,
            y0,
            x0,
            y0,
            outline="#00FF7F",
            width=2,
        )

    def _on_move(self, event: tk.Event) -> None:
        if self._canvas is None or self._rect_id is None or self._start_xy is None:
            return
        x0, y0 = self._start_xy
        x1 = int(event.x)
        y1 = int(event.y)
        self._canvas.coords(self._rect_id, x0, y0, x1, y1)

    def _on_up(self, event: tk.Event) -> None:
        if self._start_xy is None:
            self._cancel()
            return
        x0, y0 = self._start_xy
        x1, y1 = int(event.x), int(event.y)

        left = min(x0, x1) + self.vs.left
        top = min(y0, y1) + self.vs.top
        right = max(x0, x1) + self.vs.left
        bottom = max(y0, y1) + self.vs.top

        # 너무 작은 드래그는 취소로 처리
        if (right - left) < 2 or (bottom - top) < 2:
            self._cancel()
            return

        self._destroy_overlay()
        self._active = False

        self._last_result = self._capture_and_save((left, top, right, bottom))

    def _capture_and_save(self, bbox: Tuple[int, int, int, int]) -> CaptureResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_path = self.out_dir / f"capture_{ts}.png"
        meta_path = self.out_dir / f"capture_{ts}.json"

        img = ImageGrab.grab(bbox=bbox, all_screens=True)
        img.save(img_path)

        meta = {
            "bbox": {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
            "virtual_screen": {
                "left": self.vs.left,
                "top": self.vs.top,
                "width": self.vs.width,
                "height": self.vs.height,
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "image_path": str(img_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return CaptureResult(bbox=bbox, image_path=str(img_path), meta_path=str(meta_path))


class HotkeyWatcher:
    """
    pynput의 GlobalHotKeys를 사용하여 단축키를 정확하게 인식합니다.
    """

    def __init__(self, request_queue: SimpleQueue[str]) -> None:
        self.q = request_queue
        
        # 단축키 조합과 실행할 함수를 매핑
        self._listener = keyboard.GlobalHotKeys({
            '<ctrl>+<alt>+s': self._on_capture,
            '<ctrl>+<alt>+q': self._on_quit
        })

    def _on_capture(self) -> None:
        self.q.put("BEGIN_CAPTURE")

    def _on_quit(self) -> None:
        self.q.put("QUIT")

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

def main() -> int:
    out_dir = Path(__file__).resolve().parent / "captures"

    root = tk.Tk()
    root.withdraw()  # 숨긴 상태로 이벤트 루프만 사용

    q: SimpleQueue[str] = SimpleQueue()
    selector = RegionSelector(root=root, out_dir=out_dir)
    watcher = HotkeyWatcher(request_queue=q)
    watcher.start()

    print("준비됨: Ctrl+Alt+S 를 누르면 드래그 캡처 모드가 시작됩니다.")
    print(" - ESC: 캡처 취소")
    print(" - Ctrl+Alt+Q: 프로그램 종료")
    print(f" - 저장 위치: {out_dir}")

    def pump_queue() -> None:
        while True:
            try:
                msg = q.get_nowait()
            except Exception:
                break

            if msg == "BEGIN_CAPTURE":
                if not selector.active:
                    selector.begin()
            elif msg == "QUIT":
                try:
                    if selector.active:
                        selector.cancel()
                    root.quit()
                except Exception:
                    pass

        # 캡처 결과가 막 생겼으면 한 번만 출력
        res = selector.consume_last_result()
        if res is not None:
            l, t, r, b = res.bbox
            print(f"캡처 완료: bbox=({l},{t},{r},{b})")
            print(f" - 이미지: {res.image_path}")
            print(f" - 메타:  {res.meta_path}")

        root.after(30, pump_queue)

    root.after(30, pump_queue)
    try:
        try:
            root.mainloop()
        except KeyboardInterrupt:
            # Ctrl+C 등으로 종료 시 스택트레이스를 남기지 않고 조용히 종료
            pass
    finally:
        try:
            watcher.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

