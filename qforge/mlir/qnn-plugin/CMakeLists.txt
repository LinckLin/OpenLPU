add_llvm_library(QnnPlugin
        MODULE
        qnn-plugin.cpp

        DEPENDS
        MLIRQnnQisa
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRQnnQisa
        )
target_include_directories(QnnPlugin PRIVATE "${QNN_BINARY_DIR}/include")
