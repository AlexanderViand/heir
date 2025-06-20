add_heir_pass(OpenfheToScheme)

    add_mlir_conversion_library(HEIROpenfheToScheme OpenfheToScheme.cpp

                                    DEPENDS HEIROpenfheToSchemeIncGen

                                        LINK_LIBS PUBLIC HEIRRewriteUtils
                                            HEIRModuleAttributes HEIROpenfhe
                                                HEIRBGV HEIRCKKS HEIRLWE MLIRIR
                                                    MLIRPass MLIRTransformUtils)
