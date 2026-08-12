// RUN: heir-opt --cheddar-bufferize --split-input-file --verify-diagnostics %s

// Stateful destinations cannot be copied: copying a UserInterface and then
// mutating the copy would violate its unique ownership contract. The precise
// bufferization model therefore requires these destinations to be writable and
// in-place.
func.func @borrowed_ui(%ui: tensor<!cheddar.user_interface> {bufferization.writable = false}) -> tensor<!cheddar.user_interface> {
  // expected-error@+1 {{move-only read-write destination must bufferize in-place}}
  %updated = cheddar.prepare_rot_key %ui {distance = 7 : i64, maxLevel = 13 : i64} : (tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface>
  return %updated : tensor<!cheddar.user_interface>
}

// -----

// A distinct UserInterface copy is always invalid: unlike payload
// insert-slice copies, it cannot be an alias copy that a later fold removes.
func.func @materialize_move_only(
    %src: tensor<!cheddar.user_interface>,
    %dest: tensor<!cheddar.user_interface>) -> tensor<!cheddar.user_interface> {
  // expected-error @below {{bufferization cannot copy a Cheddar user interface}}
  // expected-error @below {{failed to bufferize op}}
  %result = bufferization.materialize_in_destination %src in %dest
      : (tensor<!cheddar.user_interface>, tensor<!cheddar.user_interface>)
          -> tensor<!cheddar.user_interface>
  return %result : tensor<!cheddar.user_interface>
}
