add_mlir_library(HEIRSimFHEEmitter
    SimFHEEmitter.cpp

    LINK_LIBS PUBLIC
    HEIRSelectVariableNames
    HEIRCKKS
    HEIRTargetUtils
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    MLIRTranslateLib
    LLVMSupport
)

target_link_libraries(HEIRTarget INTERFACE HEIRSimFHEEmitter)
