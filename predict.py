from ultralytics import YOLO

model = YOLO("yolo26n_best14-07-26.pt")

model.predict(
    source = "HML_DIR_MAAFUSHI.MP4", 
    show=True,
    save=True,
    conf=0.7,
    line_width=2,
    save_crop=True,
    save_txt=True,
    show_labels=True,
    show_conf=True,
    classes=[0]
    )