package main

import "core:fmt"
import "core:unicode"
import "core:strconv"

OpAssoc :: enum{NoAssoc, Left, Right}

OpIndex :: distinct int
InfixOperator :: struct {
  prec: int,
  assoc: OpAssoc
}

Token :: struct {
  token: string,
  type: TokenType,
  infix_op: Maybe(InfixOperator)
}

TokenType :: enum{Number, Ident, Paren, Comma}
NodeType :: enum{Application, Variable, Number}

ParseNode :: struct {
  tokenIndex: int,
  nodeType: NodeType,
  parentIndex: int
}

ParserState :: enum{App, Infix, NonTerminal, Terminal}

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

parseInfix :: proc(tokens: ^#soa[dynamic]Token,
                   parentNodeIndex: int,
                   currentTokenIndex: int,
                   tree_stack: ^#soa[dynamic]ParseNode,
                   minPrec: int) -> (int, int) {
  appParentIndex : int
  if len(tree_stack) > 0 {
    appParentIndex = tree_stack[len(tree_stack)-1].parentIndex
  }
  else {
    appParentIndex = -1
  }

  // parse lhs first
  fmt.println("Parsing lhs of infix", currentTokenIndex)
  fmt.println(tokens[currentTokenIndex])
  fmt.println(tokens[currentTokenIndex+1])
  assert (tokens[currentTokenIndex+1].infix_op != nil)
  nodeIndex, tokenIndex := parse(tokens, parentNodeIndex, currentTokenIndex+1, tree_stack, ParserState.Infix)

  nextMinPrec : int

  for true {
    if tokenIndex == len(tokens) {
      break
    }

    current_token := tokens[tokenIndex]

    fmt.println("current_token = ", current_token)

    if current_token.infix_op == nil {
      break
    }

    assert (current_token.infix_op != nil)

    token_prec := current_token.infix_op.?.prec
    token_assoc := current_token.infix_op.?.assoc

    fmt.println(token_prec, token_assoc)

    if token_prec < minPrec {
      break
    }

    if token_assoc == OpAssoc.Left {
      nextMinPrec = token_prec + 1
    }
    else {
      nextMinPrec = token_prec
    }

    // parse rhs
    nodeIndex, tokenIndex = parseInfix(tokens, nodeIndex, tokenIndex, tree_stack, nextMinPrec)
  }
  return nodeIndex, tokenIndex
}

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

  append(tree_stack, ParseNode{currentTokenIndex, NodeType.Application, appParentIndex})
  currentTokenIndex := currentTokenIndex+1

  assert (tokens[currentTokenIndex].token == "(")

  currentTokenIndex += 1

  currentNodeIndex := parentNodeIndex
  for currentTokenIndex < len(tokens) && tokens[currentTokenIndex].token != ")" {
    if (tokens[currentTokenIndex].token == ",") {
      currentTokenIndex += 1
    }
    currentNodeIndex, currentTokenIndex = parse(tokens, parentNodeIndex, currentTokenIndex, tree_stack, ParserState.App)
  }

  assert (tokens[currentTokenIndex].token == ")")

  return currentNodeIndex, currentTokenIndex+1
}

parse :: proc(tokens: ^#soa[dynamic]Token,
              parentNodeIndex: int,
              currentTokenIndex: int,
              tree_stack: ^#soa[dynamic]ParseNode,
              parserState: ParserState) -> (int, int) {

  fmt.println("parse started, ", tokens, parentNodeIndex, currentTokenIndex)
  // we map from parse node to token (injective function because there may be tokens that don't map back? or they could be sets of tokens)

  assert (tokens[currentTokenIndex].token != ")")
  if currentTokenIndex == len(tokens) {
    return parentNodeIndex, currentTokenIndex
  }

  token: Token = tokens[currentTokenIndex]

  switch token.type {
    case TokenType.Number:
      infixOp, infix_op_found := lookahead(tokens, currentTokenIndex, 1).?.infix_op.?

      if (infix_op_found) {
        fmt.println("Parsing an infix expression")
        newParentNodeIndex, newCurrentTokenIndex := parseInfix(tokens, len(tree_stack), currentTokenIndex, tree_stack, 0)
        return newParentNodeIndex, newCurrentTokenIndex
      }

      append(tree_stack, ParseNode{currentTokenIndex, NodeType.Number, parentNodeIndex})
      return parentNodeIndex, currentTokenIndex+1

    case TokenType.Ident:
      leftParen, left_paren_found := lookahead(tokens, currentTokenIndex, 1, "(").?
      infixOp, infix_op_found := lookahead(tokens, currentTokenIndex, 1).?.infix_op.?

      if (left_paren_found) {
        fmt.println("Parsing a function application!")

        newParentNodeIndex, newCurrentTokenIndex := parseApplication(tokens, len(tree_stack), currentTokenIndex, tree_stack)
        return newParentNodeIndex, newCurrentTokenIndex
      }
      if (infix_op_found && parserState != ParserState.Infix) { // stop parsing if an infix op is found but we're already parsing an infix expression
        fmt.println("found an infix operator identifier")
        return parentNodeIndex, currentTokenIndex
      }
      else {
        append(tree_stack, ParseNode{currentTokenIndex, NodeType.Variable, parentNodeIndex})
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
      result = OpAssoc.Left
    case "+", "-":
      result = OpAssoc.Right
  }
  return result
}

get_op :: proc(op_st : string) -> Maybe(InfixOperator) {
  if !is_operator(cast(rune)op_st[0]) {
    return nil
  }
  return InfixOperator{op_prec(op_st), op_assoc(op_st)}
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
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
    }

    if c == ',' {
      current_tok_type = TokenType.Comma
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
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
      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
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

      append(tokens, Token{string(input_st[current_tok_start:current_tok_end]), current_tok_type, nil})
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

      op_info := get_op(op_tok_st)

      append(tokens, Token{op_tok_st, current_tok_type, op_info})
    }

    current_tok_start = current_tok_end
    current_tok_end += 1
  }
}

main :: proc() {
  //test_string_app: string = "foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  test_string_infix: string = "1 + 2"
  tokens: #soa[dynamic]Token
  tree_stack: #soa[dynamic]ParseNode

  tokenize(test_string_infix, &tokens)

  for tok in tokens {
    fmt.println(tok)
  }
  nodeIndex, tokenIndex := parse(&tokens, -1, 0, &tree_stack, ParserState.NonTerminal)

  assert (tokenIndex == len(tokens))

  for node in tree_stack {
    fmt.println(node)
  }
}
