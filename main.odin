package main

import "core:fmt"
import "core:unicode"
import "core:strconv"
import "core:container/queue"

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

TokenType :: enum{Number, Ident, InfixOp, Paren, Comma}

// TODO add distinction between Variable, TypeName, and FunctionName instead of just Identifier
NodeType :: enum{Application, Identifier, Number, Root, InfixOp}

ParseNode :: struct {
  tokenIndex: int,
  nodeType: NodeType
}

ParserStates :: enum{App, NonTerminal, Terminal}

ParseState :: struct {
  nodeType: NodeType,
  tokenIndex: int,
  state: ParserStates,
  parsingInfix: bool,
  tokens: ^#soa[dynamic]Token,
  node_queue: ^queue.Queue(ParseNode),
  node_stack: ^queue.Queue(ParseNode),
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

is_operator :: proc(c : rune) -> bool {
  return (c == '*' ||
          c == '/' ||
          c == '+' ||
          c == '-' ||
          c == '^')
}

op_prec :: proc(op_st : string) -> int {
  result: int
  switch op_st {
    case "+", "-":
      result = 1
    case "*", "/":
      result = 2
    case "^":
      result = 3
  }
  return result
}

op_assoc :: proc(op_st : string) -> OpAssoc {
  result: OpAssoc
  switch op_st {
    case "*", "/", "+", "-":
      result = OpAssoc.Left
    case "^":
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
      current_tok_type = TokenType.InfixOp
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

// This will parse until it hits a terminal node
// For top-level stuff we would iterate over all the top-level non-terminals
// then recurse down to the terminal nodes the same way you would in regular RDP

// This will also check if it is inside an infix expression and in that case
// It will stop parsing once it hits an infix operator since if we are already
// doing an infix parse, we only want to parse the lhs/rhs of the expression

// another important fact is that it will push things on to the stack so
// that they can be easily evaluated later, e.g. a + b => a b +, etc
// just like in Forth

// For an example of how infix parsing will work
// f(a, g(b), c) + 134 * h
// it would parse the lhs first (in the normal parser)
// nodes for f(a, g(b), c) would get added to the stack and consumed
// we would then hit the infix operator
// start an infix parse and pass the current parser state in
// it will assume the lhs has been parsed already
// we cannot explicitly call this in the parseInfix function
// since it would parse the lhs twice basically, and there is no way to pass sub-trees around
// then we enter the infix parsing loop, and it does its thing with the precedence parsing
// then parseInfix will be called recursively for the rhs
// once it detects that the next token is not an infix op, it finishes

// also evaluation order isn't really specified so it doesn't matter if it's a b + or b a +
// they should return the same result
// if f(a) + g(b) has some side-effects then that would be
// impossible to predict which order they get evaluated in

get_parse_node :: proc(parseState: ParseState, tokenOffset: int) -> ParseNode {
  return ParseNode{parseState.tokenIndex-tokenOffset, parseState.nodeType}
}

advance_parser :: proc(parserState: ParseState,
                       nodeType: NodeType,
                       nTokensAdvance: int,
                       newState: ParserStates) -> ParseState {
  newParserState: ParseState = parserState
  newParserState.tokenIndex += nTokensAdvance
  newParserState.nodeType = nodeType
  newParserState.node_queue = parserState.node_queue
  newParserState.node_stack = parserState.node_stack
  newParserState.tokens = parserState.tokens
  newParserState.state = newState

  queue.push_back(newParserState.node_queue, get_parse_node(newParserState, nTokensAdvance))

  assert ((newParserState.tokenIndex)-nTokensAdvance <= len(newParserState.tokens))
  return newParserState
}

drain_stack :: proc(parseState: ParseState) -> ParseState {
  for queue.len(parseState.node_stack^) > 0 {
    queue.push_front(parseState.node_queue, queue.pop_back(parseState.node_stack))
  }
  return parseState
}

reset_node_type :: proc(parserState: ParseState, nodeType: NodeType) -> ParseState {
  newParserState := parserState
  queue.peek_front(newParserState.node_queue).nodeType = nodeType
  return newParserState
}

get_latest_token :: proc(parserState: ParseState) -> Maybe(Token) {
  if (parserState.tokenIndex >= len(parserState.tokens)) {
    return nil
  }
  return parserState.tokens[parserState.tokenIndex]
}

expect_token_st :: #force_inline proc(parserState: ParseState,
                                      token_value: string,
                                      lineno: int) {
  // TODO, include the line in the source code itself
  if !(parserState.tokens[parserState.tokenIndex].token == token_value) {
    fmt.panicf("Expected \"%s\" but actual token is \"%s\", line in parser = %d\n",
               token_value,
               parserState.tokens[parserState.tokenIndex].token,
               lineno)
  }
}

expect_token_type :: #force_inline proc(parserState: ParseState,
                                        token_type: TokenType,
                                        lineno: int) {
  // TODO, include the line in the source code itself
  if parserState.tokenIndex >= len(parserState.tokens) {
    return
  }
  if !(parserState.tokens[parserState.tokenIndex].type == token_type) {
    fmt.panicf("Expected token of type \"%s\" but actual type is \"%s\", line in parser = %d\n",
               token_type,
               parserState.tokens[parserState.tokenIndex].type,
               lineno)
  }
}

expect_token :: proc{expect_token_st, expect_token_type}

get_node_index :: proc(parserState: ParseState) -> int {
  return queue.len(parserState.node_stack^)-1
}

parse_sep_by :: proc(parserState: ParseState, sep: string, end: string) -> ParseState {
  fmt.println("Parsing sepby: sep =", sep, ", end =", end)
  curParserState: ParseState = parserState

  assert (curParserState.tokenIndex < len(curParserState.tokens))

  for (curParserState.tokenIndex < len(curParserState.tokens) &&
       curParserState.tokens[curParserState.tokenIndex].token != end) {
    curParserState = parse(curParserState)

    if curParserState.tokenIndex >= len(curParserState.tokens) {
      fmt.panicf("Unexpected end of source, are you missing a \"%s\" somewhere?", end)
    }

    if curParserState.tokens[curParserState.tokenIndex].token == end {
      break
    }

    expect_token(curParserState, sep, #line)
    curParserState.tokenIndex += 1
  }
  return curParserState
}

parse_infix :: proc(parserState: ParseState, minPrec: int) -> ParseState {
  fmt.println(parserState.tokens[parserState.tokenIndex:])
  assert (parserState.tokenIndex < len(parserState.tokens))
  curParserState: ParseState

  // If we just started parsing the infix expr, assume the lhs has already been parsed
  if !parserState.parsingInfix {
    curParserState = parserState
    expect_token(curParserState, TokenType.InfixOp, #line)
  }
  else {
    // otherwise parse it, and it will stop once it hits an infix op
    fmt.println("parsing lhs")
    curParserState = parse(parserState)
  }
  curParserState.parsingInfix = true

  for true {
    if curParserState.tokenIndex >= len(curParserState.tokens) {
      fmt.println("breaking because we consumed all tokens")
      break
    }

    cur_token: Token = curParserState.tokens[curParserState.tokenIndex]
    if cur_token.type != TokenType.InfixOp || cur_token.infix_op.?.prec < minPrec {
      fmt.println("breaking because the current operator has prec < minPrec, ", cur_token, minPrec)
      break
    }

    assert (cur_token.infix_op != nil)

    prec := cur_token.infix_op.?.prec
    assoc := cur_token.infix_op.?.assoc

    fmt.println(prec, assoc, cur_token.token)

    nextMinPrec: int
    if assoc == OpAssoc.Left {
      nextMinPrec = prec + 1
    }
    else {
      nextMinPrec = prec
    }

    curParserState.nodeType = NodeType.InfixOp
    infix_op := get_parse_node(curParserState, 0)

    curParserState.tokenIndex += 1

    assert (curParserState.tokens[curParserState.tokenIndex].type != TokenType.InfixOp)

    curParserState = parse_infix(curParserState, nextMinPrec)

    queue.push_back(curParserState.node_queue, infix_op)

  }
  return curParserState
}

parse_application :: proc(parserState: ParseState) -> ParseState {
  fmt.println("Parsing application")
  curParserState := parserState
  expect_token(curParserState, "(", #line)
  curParserState.tokenIndex += 1

  assert (curParserState.tokenIndex < len(curParserState.tokens))

  // Set the parent node index to the current application
  curParserState = parse_sep_by(curParserState, ",", ")")
  expect_token(curParserState, ")", #line)
  curParserState.tokenIndex += 1
  return curParserState
}

parse :: proc(parserState: ParseState) -> ParseState {

  if parserState.tokenIndex >= len(parserState.tokens) {
    fmt.println("we've consumed all tokens")
    return parserState
  }

  assert (parserState.tokens[parserState.tokenIndex].token != ")")

  newParserState: ParseState

  token: Token = parserState.tokens[parserState.tokenIndex]

  switch token.type {
    case TokenType.Number:
      // need to lookahead every time and see if the next token is an infix op??
      fmt.println("number")
      newParserState = advance_parser(parserState, NodeType.Number, 1, ParserStates.Terminal)
      return parse(newParserState)
    case TokenType.InfixOp:
      // pretty sure this will fail on e.g. 1 + f(a, 4 * 3) * 21 because it will be in infix state when it
      // parses the second argument to f, so when it encounters f it has to reset it to false
      if parserState.parsingInfix {
        // If we are already parsing an infix expression and this is an infix operator, simply return
        // the parse_infix function will handle consuming the next token
        fmt.println("in parse and there was an infix op and we were already parsing infix")
        return parserState
      }
      else {
        // The identifier represents an infix operator, so this is an infix expression
        // and since we're not already parsing an infix expression, kick off the infix parser
        fmt.println("infix application")
        newParserState = parse_infix(parserState, 1)
        return newParserState
      }
    case TokenType.Ident:
      fmt.println("identifier")
      // Consume the identifier token
      newParserState = advance_parser(parserState, NodeType.Identifier, 1, ParserStates.Terminal)
      // Check if it's a left paren, then it's a function application
      left_paren, tokens_ok := get_latest_token(newParserState).?
      if tokens_ok && left_paren.token == "(" {
        fmt.println("application")
        newParserState = reset_node_type(newParserState, NodeType.Application)
        newParserState = parse_application(newParserState)
      }
      return parse(newParserState)
    case TokenType.Paren:
      // TODO, make this start a new infix parse if it's an infix expression with parens around it
      fmt.panicf("Unexpected paren \"%s\"", token.token)
    case TokenType.Comma:
  }
  return newParserState
}

main :: proc() {
  //test_string_app: string = "foo(333*12,blarg,bar(1,2,3), aaaa, 4442, x(94, a), aad)"
  test_string: string = "1 + 111 / 2 - 4 * 99 / 4"
  //test_string: string = "a(1,2,b(5,6,h(4,22),1123, h))"
  //test_string: string = "a(1,44,g(a, 2))"
  tokens: #soa[dynamic]Token
  node_stack: queue.Queue(ParseNode)
  node_queue: queue.Queue(ParseNode)

  tokenize(test_string, &tokens)

  for tok in tokens {
    fmt.println(tok)
  }

  parseState := parse(ParseState{NodeType.Root, 0, ParserStates.NonTerminal, false, &tokens, &node_queue, &node_stack})

  //assert (parseState.tokenIndex == len(tokens))

  for queue.len(node_queue) > 0 {
    node := queue.pop_front(&node_queue)
    fmt.println(tokens[node.tokenIndex])
  }
}
