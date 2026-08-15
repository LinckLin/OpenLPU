add_mlir_dialect_library(MLIRQnnQisa
        QnnDialect.cpp
        QnnOps.cpp
        QisaDialect.cpp
        QisaOps.cpp
        QnnToQisa.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/QNN

        DEPENDS
        MLIRQnnOpsIncGen
        MLIRQisaOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        MLIRPass
        MLIRSupport
        )
