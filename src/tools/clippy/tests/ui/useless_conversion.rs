// run-rustfix

#![deny(clippy::useless_conversion)]

fn test_generic<T: Copy>(val: T) -> T {
    let _ = T::from(val);
    val.into()
}

fn test_generic2<T: Copy + Into<i32> + Into<U>, U: From<T>>(val: T) {
    // ok
    let _: i32 = val.into();
    let _: U = val.into();
    let _ = U::from(val);
}

fn test_questionmark() -> Result<(), ()> {
    {
        let _: i32 = 0i32.into();
        Ok(Ok(()))
    }??;
    Ok(())
}

fn test_issue_3913() -> Result<(), std::io::Error> {
    use std::fs;
    use std::path::Path;

    let path = Path::new(".");
    for _ in fs::read_dir(path)? {}

    Ok(())
}

fn test_issue_5833() -> Result<(), ()> {
    let text = "foo\r\nbar\n\nbaz\n";
    let lines = text.lines();
    if Some("ok") == lines.into_iter().next() {}

    Ok(())
}

fn main() {
    test_generic(10i32);
    test_generic2::<i32, i32>(10i32);
    test_questionmark().unwrap();
    test_issue_3913().unwrap();
    test_issue_5833().unwrap();

    let _: String = "foo".into();
    let _: String = From::from("foo");
    let _ = String::from("foo");
    #[allow(clippy::useless_conversion)]
    {
        let _: String = "foo".into();
        let _ = String::from("foo");
        let _ = "".lines().into_iter();
    }

    let _: String = "foo".to_string().into();
    let _: String = From::from("foo".to_string());
    let _ = String::from("foo".to_string());
    let _ = String::from(format!("A: {:04}", 123));
    let _ = "".lines().into_iter();
    let _ = vec![1, 2, 3].into_iter().into_iter();
    let _: String = format!("Hello {}", "world").into();

    // keep parenthesis around `a + b` for suggestion (see #4750)
    let a: i32 = 1;
    let b: i32 = 1;
    let _ = i32::from(a + b) * 3;
}
