Heavily based on ideas from (this talk on the Carbon compiler)[https://www.youtube.com/watch?v=ZI198eFghJk] as well as other ideas around Data-Oriented Design.
The idea is to avoid using any sort of nested AST design as is traditionally done using structs with child pointers, etc. Instead it just uses arrays which for the parse tree
end up being in a specific ordering which allows you to parse it, and in my implementation currently child nodes "link" back to their parents using an index which is stored in a separate column.
