import threading


def play_alert():
    def sound():
        try:
            import winsound
            winsound.Beep(1200, 500)
            winsound.Beep(1500, 500)
            winsound.Beep(1200, 500)
        except Exception:
            print("\\aFALL ALERT!")

    threading.Thread(target=sound, daemon=True).start()
