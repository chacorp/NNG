"""NNG desktop app: shows the node-graph viewer in a native Qt window (bundled Chromium)."""
import argparse
import base64
import json
import pathlib
import sys
import threading

from PySide6.QtCore import QFile, QIODevice, QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineScript
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow

from nng_app.torch_check import run_generated_model_check

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_UI_PATH = PROJECT_ROOT / "ui" / "index.html"
BROWSER_STORAGE_PATH = pathlib.Path.home() / ".nng" / "webengine"
FILE_TYPE_FILTERS = {
    "python": "Python file (*.py)",
    "json": "Graph JSON (*.json)",
    "svg": "SVG image (*.svg)",
    "png": "PNG image (*.png)",
}
BRIDGE_BOOTSTRAP_JS = """
new QWebChannel(qt.webChannelTransport, function(channel){
  const backend = channel.objects.backend;
  const pending_checks = {};
  let next_request_id = 0;
  backend.torch_check_finished.connect(function(request_id, result_json){
    const resolve = pending_checks[request_id]; delete pending_checks[request_id];
    if(resolve) resolve(JSON.parse(result_json));
  });
  const call = (method, ...args) => new Promise(resolve => backend[method](...args, result => resolve(result ? JSON.parse(result) : null)));
  window.nng_bridge = {
    save_text_file: (file_name, file_text, file_kind) => call('save_text_file', file_name, file_text, file_kind),
    save_png_file: (file_name, png_base64) => call('save_png_file', file_name, png_base64),
    open_graph_file: () => call('open_graph_file'),
    run_torch_check: model_code => new Promise(resolve => {
      const request_id = String(next_request_id++); pending_checks[request_id] = resolve;
      backend.start_torch_check(request_id, model_code);
    }),
  };
  window.dispatchEvent(new Event('nngbridgeready'));
});
"""


class ViewerBridge(QObject):
    """Functions the viewer page calls through window.nng_bridge.* (each returns a Promise in JS).

    Slots return JSON strings; the bootstrap script parses them for the page.
    """

    torch_check_finished = Signal(str, str)

    def __init__(self, main_window):
        """Create the bridge.

        Args:
            main_window (QMainWindow): parent window for native file dialogs.
        """
        super().__init__()
        self._main_window = main_window

    @Slot(str, str, str, result=str)
    def save_text_file(self, suggested_file_name, file_text, file_kind):
        """Ask where to save and write a text file (model.py, graph JSON, SVG figure).

        Args:
            suggested_file_name (str): name pre-filled in the save dialog.
            file_text (str): file contents.
            file_kind (str): one of "python", "json", "svg".

        Returns:
            str: JSON of the saved path, or JSON null when the dialog was cancelled.
        """
        save_path = self._ask_save_path(suggested_file_name, file_kind)
        if save_path is None:
            return json.dumps(None)
        save_path.write_text(file_text, encoding="utf-8")
        return json.dumps(str(save_path))

    @Slot(str, str, result=str)
    def save_png_file(self, suggested_file_name, png_base64):
        """Ask where to save and write a PNG figure.

        Args:
            suggested_file_name (str): name pre-filled in the save dialog.
            png_base64 (str): PNG bytes encoded as base64 (no data: prefix).

        Returns:
            str: JSON of the saved path, or JSON null when the dialog was cancelled.
        """
        save_path = self._ask_save_path(suggested_file_name, "png")
        if save_path is None:
            return json.dumps(None)
        save_path.write_bytes(base64.b64decode(png_base64))
        return json.dumps(str(save_path))

    @Slot(result=str)
    def open_graph_file(self):
        """Ask for a graph JSON file and read it.

        Returns:
            str: JSON {"path": str, "text": str}, or JSON null when the dialog was cancelled.
        """
        chosen_path, _ = QFileDialog.getOpenFileName(self._main_window, "그래프 열기", "", FILE_TYPE_FILTERS["json"])
        if not chosen_path:
            return json.dumps(None)
        graph_path = pathlib.Path(chosen_path)
        return json.dumps({"path": str(graph_path), "text": graph_path.read_text(encoding="utf-8")})

    @Slot(str, str)
    def start_torch_check(self, request_id, model_code):
        """Run the exported model code in the background; emits torch_check_finished when done.

        Args:
            request_id (str): id echoed back so the page can match the answer.
            model_code (str): full text of the exported model.py.
        """
        def run_and_report():
            check_result = run_generated_model_check(model_code)
            self.torch_check_finished.emit(request_id, json.dumps(check_result))

        threading.Thread(target=run_and_report, daemon=True).start()

    def _ask_save_path(self, suggested_file_name, file_kind):
        """Show the native save dialog.

        Args:
            suggested_file_name (str): name pre-filled in the dialog.
            file_kind (str): key of FILE_TYPE_FILTERS.

        Returns:
            pathlib.Path | None: chosen path, or None when cancelled.
        """
        chosen_path, _ = QFileDialog.getSaveFileName(
            self._main_window, "저장", suggested_file_name, FILE_TYPE_FILTERS[file_kind]
        )
        return pathlib.Path(chosen_path) if chosen_path else None


def read_qt_resource_text(resource_path):
    """Read a text file bundled in Qt resources.

    Args:
        resource_path (str): resource path, e.g. ":/qtwebchannel/qwebchannel.js".

    Returns:
        str: file contents.
    """
    resource_file = QFile(resource_path)
    resource_file.open(QIODevice.OpenModeFlag.ReadOnly)
    resource_text = bytes(resource_file.readAll()).decode("utf-8")
    resource_file.close()
    return resource_text


def build_main_window(ui_path):
    """Create the app window with the viewer page and the Python bridge attached.

    Args:
        ui_path (pathlib.Path): viewer HTML file.

    Returns:
        tuple[QMainWindow, QWebEngineView, ViewerBridge]: window, web view, bridge.
    """
    main_window = QMainWindow()
    main_window.setWindowTitle("NNG — Neural Network Graph")
    main_window.resize(1440, 900)

    BROWSER_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    browser_profile = QWebEngineProfile("nng", QApplication.instance())
    browser_profile.setPersistentStoragePath(str(BROWSER_STORAGE_PATH))

    bridge_script = QWebEngineScript()
    bridge_script.setName("nng_bridge")
    bridge_script.setSourceCode(read_qt_resource_text(":/qtwebchannel/qwebchannel.js") + BRIDGE_BOOTSTRAP_JS)
    bridge_script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
    bridge_script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    browser_profile.scripts().insert(bridge_script)

    web_page = QWebEnginePage(browser_profile, main_window)
    bridge = ViewerBridge(main_window)
    web_channel = QWebChannel(web_page)
    web_channel.registerObject("backend", bridge)
    web_page.setWebChannel(web_channel)

    web_view = QWebEngineView(main_window)
    web_view.setPage(web_page)
    web_view.load(QUrl.fromLocalFile(str(ui_path)))
    main_window.setCentralWidget(web_view)
    return main_window, web_view, bridge


def run_self_test(web_view, application):
    """Export every example in the loaded page, torch-check each one, print a summary and quit.

    Args:
        web_view (QWebEngineView): view with the loaded viewer page.
        application (QApplication): app to quit when finished.
    """
    example_names = ["mlp", "resnet", "unet", "transformer"]
    test_report = {"all_passed": True}

    def check_next_example(example_index):
        if example_index == len(example_names):
            web_view.page().runJavaScript("!!window.nng_bridge", 0, finish)
            return
        example_name = example_names[example_index]

        def on_export(export_json):
            exported = json.loads(export_json)
            if "code" not in exported:
                print(f"[{example_name}] export failed: {exported.get('problems')}", flush=True)
                test_report["all_passed"] = False
            else:
                check_result = run_generated_model_check(exported["code"])
                ok_line = next((line for line in check_result["output"].splitlines() if line.startswith("OK:")), "")
                print(f"[{example_name}] {'PASS' if check_result['passed'] else 'FAIL'}  {ok_line}", flush=True)
                test_report["all_passed"] = test_report["all_passed"] and check_result["passed"]
            check_next_example(example_index + 1)

        web_view.page().runJavaScript(
            f"load_example({json.dumps(example_name)}); JSON.stringify(generate_model_code())", 0, on_export
        )

    def finish(bridge_ready):
        print(f"bridge visible to the page: {bridge_ready}", flush=True)
        print("SELF TEST", "PASSED" if test_report["all_passed"] and bridge_ready else "FAILED", flush=True)
        application.quit()

    check_next_example(0)


def main():
    """Start the NNG window.

    Command line:
        python -m nng_app [--ui PATH] [--self-test]
    """
    argument_parser = argparse.ArgumentParser(description="NNG neural-network graph designer")
    argument_parser.add_argument("--ui", type=pathlib.Path, default=DEFAULT_UI_PATH, help="viewer HTML file")
    argument_parser.add_argument("--self-test", action="store_true", help="export every example, torch-check it, then quit")
    arguments = argument_parser.parse_args()

    ui_path = arguments.ui.resolve()
    if not ui_path.exists():
        sys.exit(f"Viewer page not found: {ui_path}")

    application = QApplication(sys.argv)
    main_window, web_view, _bridge = build_main_window(ui_path)
    if arguments.self_test:
        web_view.loadFinished.connect(lambda loaded_ok: run_self_test(web_view, application))
    main_window.show()
    sys.exit(application.exec())
