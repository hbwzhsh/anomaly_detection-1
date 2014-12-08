QMAKE_LFLAGS += -D__STDC_CONSTANT_MACROS
LIBS += -lc -lavcodec -lavformat -lavutil -lswscale

INCLUDEPATH += /usr/local/include
LIBS += -L/usr/loca/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_nonfree -lopencv_features2d -lopencv_calib3d

HEADERS += \
    timing.h \
    options.h \
    motion_vector_file_utils.h \
    log.h \
    io_utils.h \
    integral_transform.h \
    histogram_buffer.h \
    frame_reader.h \
    diag.h \
    desc_info.h \
    common.h \
    residual.h \
    interaction_energy.h

SOURCES += \
    main.cpp
