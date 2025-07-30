#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEENUMS_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEENUMS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"

include "lib/Dialect/Openfhe/IR/OpenfheDialect.td"

def Openfhe_ScalingTechniqueEnum : I32Enum<"ScalingTechniqueEnum",
    "An enum attribute representing OpenFHE scaling techniques",
    [ I32EnumCase<"FIXEDMANUAL", 0>,
      I32EnumCase<"FIXEDAUTO", 1>,
      I32EnumCase<"FLEXIBLEAUTO", 2>,
      I32EnumCase<"FLEXIBLEAUTOEXT", 3>,
      I32EnumCase<"NORESCALE", 4>,
      I32EnumCase<"COMPOSITESCALINGAUTO", 5>
    ]>

{
  let cppNamespace = "::mlir::heir::openfhe";
}

def Openfhe_ScalingTechniqueEnumAttr : EnumAttr<Openfhe_Dialect, Openfhe_ScalingTechniqueEnum, "ScalingTechnique">;

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEENUMS_TD_
