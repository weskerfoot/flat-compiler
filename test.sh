#! /usr/bin/env bash

tests=(
  #"foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  #"1 + 111 / (2 - (4 +5)) *(99/ 4)"
  #"a := 1 + 23 + 2 * 3"
  #"1 * 2 + 12 * cos((3 / 4) - 14)"
  #"cos(12 + 4) a(1,2)"
  #"foobar := sin(14 + 12) * cos(2 - 3); a + b * c"
  #"2 + 3 * -12 / 2"
  "2 + 3 * 12 - 2"
  #"2 + 13 * f(a, 3)  / 12;"
)

make build

for i in "${!tests[@]}"; do
  echo "Running test $((i + 1)):"
  echo "Input = ${tests[$i]}"
  echo "${tests[$i]}" | ./flat-compiler
  echo "-------------------------"
done

