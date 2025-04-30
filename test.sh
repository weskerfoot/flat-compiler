test_1="foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
test_2="1 + 111 / (2 - (4 +5)) *(99/ 4)"
test_3="a := 1 + 23 + 2 * 3"
test_4="1 * 2 + 12 * cos((3 / 4) - 14)"
test_5="cos(12 + 4) a(1,2)"
test_6="foobar := sin(14 + 12) * cos(2 - 3); a + b * c"

make build

echo "Test input = $test_1"
echo "$test_1" | ./flat-compiler
echo "Test input = $test_2"
echo "$test_2" | ./flat-compiler
echo "Test input = $test_3"
echo "$test_3" | ./flat-compiler
