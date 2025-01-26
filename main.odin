package main

import "core:fmt"
import "core:unicode"
import "core:strconv"

OpAssoc :: enum{NoAssoc, LeftAssoc, RightAssoc}

Token :: struct {
  token: string,
  type: TokenType,
  prec: int,
  assoc: OpAssoc
}

TokenType :: enum{Number, Ident, Paren, Comma}
NodeType :: enum{Application, Variable, Number}

ParseNode :: struct {
  tokenIndex: int,
  nodeType: NodeType,
  childIndex: int, // if there's multiple children, it's the "last" child's index which should end up being laid out in order on the stack
  parentIndex: int
}

lookahead_basic :: proc(tokens: ^#soa[dynamic]Token,
                        start: int,
                        n: int) -> Maybe(Token) {
  if start+n >= len(tokens) {
    return nil
  }
  return tokens[start+n]
}

lookahead_with_tokvalue :: proc(tokens: ^#soa[dynamic]Token,
                                start: int,
                                n: int,
                                tokValue: string) -> Maybe(Token) {
  if start+n >= len(tokens) {
    return nil
  }
  result := tokens[start+n]
  if (result.token != tokValue) {
    return nil
  }
  return result
}

lookahead :: proc{lookahead_with_tokvalue, lookahead_basic}

parseApplication :: proc(tokens: ^#soa[dynamic]Token,
                         parentNodeIndex: int,
                         currentTokenIndex: int,
                         tree_stack: ^#soa[dynamic]ParseNode) -> (int, int) {
  appParentIndex : int
  if len(tree_stack) > 0 {
    appParentIndex = tree_stack[len(tree_stack)-1].parentIndex
  }
  else {
    appParentIndex = -1
  }

  append(tree_stack, ParseNode{currentTokenIndex, NodeType.Application, len(tree_stack)-1, appParentIndex})
  currentTokenIndex := currentTokenIndex+1

  assert (tokens[currentTokenIndex].token == "(")

  currentTokenIndex += 1

  currentNodeIndex := parentNodeIndex
  for currentTokenIndex < len(tokens) && tokens[currentTokenIndex].token != ")" {
    if (tokens[currentTokenIndex].token == ",") {
      currentTokenIndex += 1
    }
    currentNodeIndex, currentTokenIndex = parse(tokens, parentNodeIndex, currentTokenIndex, tree_stack)
  }

  assert (tokens[currentTokenIndex].token == ")")

  return currentNodeIndex, currentTokenIndex+1
}

parse :: proc(tokens: ^#soa[dynamic]Token,
              parentNodeIndex: int,
              currentTokenIndex: int,
              tree_stack: ^#soa[dynamic]ParseNode) -> (int, int) {

  // we map from parse node to token (injective function because there may be tokens that don't map back? or they could be sets of tokens)

  assert (tokens[currentTokenIndex].token != ")")
  if currentTokenIndex == len(tokens) {
    return parentNodeIndex, currentTokenIndex
  }

  token: Token = tokens[currentTokenIndex]

  switch token.type {
    case TokenType.Number:

      append(tree_stack, ParseNode{currentTokenIndex, NodeType.Number, len(tree_stack)-1, parentNodeIndex})
      return parentNodeIndex, currentTokenIndex+1

    case TokenType.Ident:
      leftParen, left_paren_found := lookahead(tokens, currentTokenIndex, 1, "(").?
      if (left_paren_found) {
        fmt.println("Parsing a function application!")

        newParentNodeIndex, newCurrentTokenIndex := parseApplication(tokens, len(tree_stack), currentTokenIndex, tree_stack)
        return newParentNodeIndex, newCurrentTokenIndex
      }
      else {
        append(tree_stack, ParseNode{currentTokenIndex, NodeType.Variable, len(tree_stack)-1, parentNodeIndex})
        return parentNodeIndex, currentTokenIndex+1
      }
    case TokenType.Paren:
    case TokenType.Comma:
  }
  return parentNodeIndex, currentTokenIndex
}



is_operator :: proc(c : rune) -> bool {
  return (c == '*' ||
          c == '/' ||
          c == '+' ||
          c == '-')
}

op_prec :: proc(op_st : string) -> int {
  result: int
  switch op_st {
    case "*", "/":
      result = 3
    case "+", "-":
      result = 2
  }
  return result
}

op_assoc :: proc(op_st : string) -> OpAssoc {
  result: OpAssoc
  switch op_st {
    case "*", "/":
      result = OpAssoc.LeftAssoc
    case "+", "-":
      result = OpAssoc.RightAssoc
  }
  return result
}

is_identifier_char :: proc(c : rune) -> bool {
  return !is_operator(c) && !unicode.is_number(c) && !unicode.is_space(c) && c != '(' && c != ')' && c != ','
}

tokenize :: proc(input_st: string, tokens: ^#soa[dynamic]Token) {
  current_tok_start: int = 0
  current_tok_end: int = 1

  current_tok_type: TokenType

  for current_tok_start < len(input_st) {
    c := rune(input_st[current_tok_start])

    if c == '(' || c == ')' {
      current_tok_type = TokenType.Paren
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, 0, OpAssoc.NoAssoc})
    }

    if c == ',' {
      current_tok_type = TokenType.Comma
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, 0, OpAssoc.NoAssoc})
    }

    if unicode.is_number(c) {
      current_tok_type = TokenType.Number

      for current_tok_end < len(input_st) {
        next_digit := rune(input_st[current_tok_end])
        if current_tok_end > len(input_st) || !unicode.is_number(next_digit) {
          break
        }
        current_tok_end += 1
      }
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, 0, OpAssoc.NoAssoc})
    }

    if is_identifier_char(c) {
      current_tok_type = TokenType.Ident
      for current_tok_end < len(input_st) {
        next_letter := rune(input_st[current_tok_end])

        if current_tok_end >= len(input_st) || !is_identifier_char(next_letter) {
          break
        }
        current_tok_end += 1
      }
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, 1, OpAssoc.NoAssoc})
    }

    if is_operator(c) {
      current_tok_type = TokenType.Ident
      for current_tok_end < len(input_st) {
        next_letter := rune(input_st[current_tok_end])

        if current_tok_end >= len(input_st) || !is_identifier_char(next_letter) {
          break
        }
        current_tok_end += 1
      }

      op_tok_st: string = string(input_st[current_tok_start:current_tok_end])

      append(tokens, Token{op_tok_st, current_tok_type, op_prec(op_tok_st), op_assoc(op_tok_st)})
    }

    current_tok_start = current_tok_end
    current_tok_end += 1
  }
}

main :: proc() {
  //test_string: string = "(12 + (4 * 3) / 234)*(-1555) + abc"
  //test_string: string = "123 4 333 abcd(44, 23, foobar)"
  test_string: string = "foo(333,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  tokens: #soa[dynamic]Token
  tree_stack: #soa[dynamic]ParseNode

  tokenize(test_string, &tokens)

  for tok in tokens {
    fmt.println(tok)
  }
  nodeIndex, tokenIndex := parse(&tokens, -1, 0, &tree_stack)

  assert (tokenIndex == len(tokens))

  for node in tree_stack {
    fmt.println(node)
  }
}
