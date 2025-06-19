from heir import compile_mlir
from heir.backends.cleartext import CleartextBackend
from absl.testing import absltest


class CompileMlirTest(absltest.TestCase):

  def test_compile_mlir(self):
    mlir = (
        "func.func @foo(%x: i16 {secret.secret}, %y: i16 {secret.secret}) -> (i16) {\n"
        "  %0 = arith.addi %x, %y : i16\n"
        "  func.return %0 : i16\n"
        "}\n"
    )
    client = compile_mlir(mlir, backend=CleartextBackend())
    self.assertIsNotNone(client)


if __name__ == "__main__":
  absltest.main()
