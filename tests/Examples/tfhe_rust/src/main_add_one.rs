use tfhe::shortint::prelude::*;

// Encrypt a u8
pub fn encrypt(value: u8, client_key: &ClientKey) -> [Ciphertext; 8] {
    core::array::from_fn(|shift| {
        let bit = (value >> shift) & 1;
        client_key.encrypt(if bit != 0 { 1 } else { 0 })
    })
}

// Decrypt a u8
pub fn decrypt(ciphertexts: &[Ciphertext], client_key: &ClientKey) -> u8 {
    let mut accum = 0u8;
    for (i, ct) in ciphertexts.iter().enumerate() {
        let bit = client_key.decrypt(ct);
        accum |= (bit as u8) << i;
    }
    accum
}

#[cfg(test)]
mod test {
    use tfhe::shortint::prelude::*;

    use super::decrypt;
    use super::encrypt;

    use add_one_test_rs_lib;

    #[test]
    fn simple_test() {
        let (client_key, server_key) = gen_keys(PARAM_MESSAGE_2_CARRY_2_KS_PBS);

        let ct_1 = encrypt(3, &client_key);

        let result = add_one_test_rs_lib::fn_under_test(&server_key, &ct_1);
        let output = decrypt(&result, &client_key);

        assert_eq!(output, 4);
    }
}
