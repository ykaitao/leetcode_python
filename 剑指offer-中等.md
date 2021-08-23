# [剑指 Offer 04. 二维数组中的查找](https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/)
在一个 n * m 的二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个高效的函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

 

示例:

现有矩阵 matrix 如下：
```
[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
```
给定 target = 5，返回 true。

给定 target = 20，返回 false。

代码：
```python3
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:

        m = len(matrix) # number of rows
        if m==0: return False # matrix = []
        n = len(matrix[0]) # number of columns

        i, j = 0, n-1 # right-top
        while i<m and j>-1:
            # if found
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        
        return False
```

# [剑指 Offer 12. 矩阵中的路径](https://leetcode-cn.com/problems/ju-zhen-zhong-de-lu-jing-lcof/)
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。

单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
 

示例 1：
```
输入：board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
输出：true
```
示例 2：
```
输入：board = [["a","b"],["c","d"]], word = "abcd"
输出：false
```

代码：

```python3
class Solution:

    def exist(self, board: List[List[str]], word: str) -> bool:

        def dfs(i: int, j: int, k: int=0)->bool:
            # check boundary
            if i<0 or i>=m or j<0 or j>=n: return False
            # check re-visit
            if board[i][j] == "": return False
            # doese not match with word
            if board[i][j] != word[k]: return False
            # check found
            if k == len(word)-1: return True

            # mark as visited
            board[i][j] = ""
            
            if (
                dfs(i, j-1, k+1) or # move left
                dfs(i, j+1, k+1) or # move right
                dfs(i-1, j, k+1) or # move top
                dfs(i+1, j, k+1) # move bottom
            ): return True
            
            board[i][j] = word[k] # remove mark

        m = len(board)
        if m==0: return False
        n = len(board[0])
        if n==0: return False
        for i in range(m):
            for j in range(n):
                if dfs(i, j, k=0):
                    return True
        return False
```

# [剑指 Offer 13. 机器人的运动范围](https://leetcode-cn.com/problems/ji-qi-ren-de-yun-dong-fan-wei-lcof/)
地上有一个m行n列的方格，从坐标 [0,0] 到坐标 [m-1,n-1] 。一个机器人从坐标 [0, 0] 的格子开始移动，它每次可以向左、右、上、下移动一格（不能移动到方格外），也不能进入行坐标和列坐标的数位之和大于k的格子。例如，当k为18时，机器人能够进入方格 [35, 37] ，因为3+5+3+7=18。但它不能进入方格 [35, 38]，因为3+5+3+8=19。请问该机器人能够到达多少个格子？

 

示例 1：
```
输入：m = 2, n = 3, k = 1
输出：3
```
示例 2：
```
输入：m = 3, n = 1, k = 0
输出：1
```
提示：
```
1 <= n,m <= 100
0 <= k <= 20
```

代码：

```python3
class Solution:
    def movingCount(self, m: int, n: int, k: int) -> int:
        def sum(i: int) -> int:
            s = 0
            while i>0:
                s += i%10
                i = i // 10
            return s 

        def dfs(i: int, j:int) -> int:
            # check boundary
            if i>=m or j>=n: return 0
            # check re-visit
            if (i, j) in self.visited: return 0
            # check sum
            if sum(i)+sum(j)>k: return 0
            

            # mark visited
            self.visited.add((i, j))
            return (
                1
                + dfs(i, j+1) # move right
                + dfs(i+1, j) # move bottom
            )
        
        self.visited = set()
        return dfs(i=0, j=0)
```

# [剑指 Offer 07. 重建二叉树](https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof/)

输入某二叉树的前序遍历和中序遍历的结果，请构建该二叉树并返回其根节点。

假设输入的前序遍历和中序遍历的结果中都不含重复的数字。

 

示例 1:
```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

示例 2:
```
Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

限制：
```
0 <= 节点个数 <= 5000
```

代码：

```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:

        def recursive_build(
            p_start: int = 0, p_end: int = None,
            i_start: int = 0, i_end: int = None
        ) -> TreeNode:
            if p_end is None: p_end = len(preorder)
            if i_end is None: i_end = len(inorder)
            
            if p_start==p_end:
                return

            x = preorder[p_start] 
            i = self.dic[x] # find index of x in inorder
            lenl, lenr = i - i_start, i_end - i - 1 # length of left, right trees
            root = TreeNode(val=x)
            root.left = recursive_build(
                p_start=p_start+1, p_end=p_start+1+lenl,
                i_start=i_start, i_end=i,
            ) 
            root.right = recursive_build(
                p_start=p_start+1+lenl, p_end=p_start+1+lenl+lenr,
                i_start=i+1, i_end=i_end,
            )
            return root

        self.dic = {val: idx for idx, val in enumerate(inorder)}
        return recursive_build()
```
# [剑指 Offer 14- I. 剪绳子](https://leetcode-cn.com/problems/jian-sheng-zi-lcof/)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m-1] 。请问 k[0]*k[1]*...*k[m-1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

示例 1：
```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```
示例 2:
```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```
提示：
```
2 <= n <= 58
```

代码：

```python3
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n==2: return 1
        if n==3: return 2

        n_segments = n // 3 # n_segments>=1
        rem = n % 3
        
        if rem==0:
            return 3**n_segments
        elif rem==1:
            return 3**(n_segments-1) * (3-1) * (rem+1)
        else:
            return 3**n_segments*2
```
# [剑指 Offer 14- II. 剪绳子 II](https://leetcode-cn.com/problems/jian-sheng-zi-ii-lcof/)

给你一根长度为 n 的绳子，请把绳子剪成整数长度的 m 段（m、n都是整数，n>1并且m>1），每段绳子的长度记为 k[0],k[1]...k[m - 1] 。请问 k[0]*k[1]*...*k[m - 1] 可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：
```
输入: 2
输出: 1
解释: 2 = 1 + 1, 1 × 1 = 1
```

示例 2:
```
输入: 10
输出: 36
解释: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36
```

提示：
```
2 <= n <= 1000
```

代码：

```python3
class Solution:
    def cuttingRope(self, n: int) -> int:
        if n==2: return 1
        if n==3: return 2

        n_three = n//3 # n_three>=1
        rem = n % 3

        if rem==0:
            ans = 3**n_three
        elif rem==1:
            ans = 3**(n_three-1) * 4
        else:
            ans = 3**n_three * 2
        return ans % 1000000007
```

# [剑指 Offer 26. 树的子结构](https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof/)

输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:
```
     3
    / \
   4   5
  / \
 1   2
```
给定的树 B：
```
   4 
  /
 1
```
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：
```
输入：A = [1,2,3], B = [3,1]
输出：false
```
示例 2：
```
输入：A = [3,4,5,1,2], B = [4,1]
输出：true
```
限制：
```
0 <= 节点个数 <= 10000
```
代码：

```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        
        def is_subtree_from_root(A: TreeNode, B: TreeNode) -> bool:
            if B is None: return True
            if A is None: return False
            if A.val != B.val: return False
            return (
                is_subtree_from_root(A.left, B.left) and 
                is_subtree_from_root(A.right, B.right)
            )
        
        if A is None or B is None: return False
        return (
            is_subtree_from_root(A, B) or
            self.isSubStructure(A.left, B) or
            self.isSubStructure(A.right, B)
        )
```

# [剑指 Offer 20. 表示数值的字符串](https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof/)

请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。

数值（按顺序）可以分成以下几个部分：

若干空格
一个 小数 或者 整数
（可选）一个 'e' 或 'E' ，后面跟着一个 整数
若干空格
小数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
下述格式之一：
至少一位数字，后面跟着一个点 '.'
至少一位数字，后面跟着一个点 '.' ，后面再跟着至少一位数字
一个点 '.' ，后面跟着至少一位数字
整数（按顺序）可以分成以下几个部分：

（可选）一个符号字符（'+' 或 '-'）
至少一位数字
部分数值列举如下：
```
["+100", "5e2", "-123", "3.1416", "-1E-16", "0123"]
```
部分非数值列举如下：
```
["12e", "1a3.14", "1.2.3", "+-5", "12e+5.4"]
 
```
示例 1：
```
输入：s = "0"
输出：true
```
示例 2：
```
输入：s = "e"
输出：false
```
示例 3：
```
输入：s = "."
输出：false
```
示例 4：
```
输入：s = "    .1  "
输出：true
```

提示：
```
1 <= s.length <= 20
s 仅含英文字母（大写和小写），数字（0-9），加号 '+' ，减号 '-' ，空格 ' ' 或者点 '.' 。
```
代码：

```python3
class Solution:
    def isNumber(self, s: str) -> bool:
        valid_chars = set(" +-0123456789eE.")

        state_trans = [
            {" ": 0, "+-": 2, "0123456789":4, ".": 8},  # "0 space start",
            {" ": 1},                                   # "1 space end",
            {"0123456789": 4, ".": 8},                  # "2 sign",
            {"0123456789": 6},                          # "3 sign after eE",
            {"0123456789": 4, "eE": 9, ".": 7, " ": 1}, # "4 digit before .",
            {"eE": 9, "0123456789": 5, " ": 1},         # "5 digit after .",
            {"0123456789": 6, " ": 1},                  # "6 digit after eE",
            {"0123456789": 5, " ":1, "eE": 9},          # "7 . with left digit",
            {"0123456789": 5},                          # "8 . without left digit",
            {"+-": 3, "0123456789": 6},                 # "9 eE",
        ]

        # iterate over string
        curr_state = 0
        for c in s:
            if c not in valid_chars:
                return False

            next_state = None
            for key, val in state_trans[curr_state].items():
                if c in key:
                    next_state = val
            if next_state is None:
                return False
            curr_state = next_state

        return next_state in [1, 4, 5, 6, 7]
```
# [剑指 Offer 16. 数值的整数次方](https://leetcode-cn.com/problems/shu-zhi-de-zheng-shu-ci-fang-lcof/)

实现 `pow(x, n)` ，即计算 x 的 n 次幂函数（即，xn）。不得使用库函数，同时不需要考虑大数问题。

 

示例 1：
```
输入：x = 2.00000, n = 10
输出：1024.00000
```
示例 2：
```
输入：x = 2.10000, n = 3
输出：9.26100
```
示例 3：
```
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25
```

提示：
```
-100.0 < x < 100.0
-231 <= n <= 231-1
-104 <= xn <= 104
```
代码：

```python3
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x==0 or x==1: return x
        if n<0: return self.myPow(1/x, -n)
        if n==0: return 1

        ans = 1
        while n>=1:
            if n%2==1:
                # if n is old
                ans = ans * x

            x = x**2
            n = n>>1
             
        return ans
```
# [剑指 Offer 35. 复杂链表的复制](https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof/)

请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。


示例 1：
```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```
示例 2：
```
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
```
示例 3：
```
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
```
示例 4：
```
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。
```

提示：
```
-10000 <= Node.val <= 10000
Node.random 为空（null）或指向链表中的节点。
节点数目不超过 1000 。
```

代码：

```python3
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head is None: return

        # copy
        old_node = head
        while old_node:
            new_node = Node(old_node.val)
            new_node.next = old_node.next
            old_node.next = new_node
            old_node = new_node.next

        # assign random
        old_node = head
        while old_node:
            new_node = old_node.next
            # Pay attention, the random of old node could be null
            if old_node.random:
                new_node.random = old_node.random.next
            old_node = new_node.next

        # split
        new_head = new_node = head.next
        while new_node.next:
            next_old_node = new_node.next
            new_node.next = next_old_node.next
            new_node = next_old_node.next
        return new_head
```

# [剑指 Offer 36. 二叉搜索树与双向链表](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/)

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：

<img src=https://assets.leetcode.com/uploads/2018/10/12/bstdlloriginalbst.png width="700"/>

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

<img src=https://assets.leetcode.com/uploads/2018/10/12/bstdllreturndll.png width="700"/>
 

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。



代码：

```python3
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        
        def travel_inorder(node_curr: 'Node') -> None:
            if node_curr is None: return

            # travel left
            travel_inorder(node_curr.left)

            # do sth
            node_curr.left = self.node_pred
            if self.node_pred:
                self.node_pred.right = node_curr
            self.node_pred = node_curr
            if self.head is None:
                self.head = node_curr

            # travel right
            travel_inorder(node_curr.right)
        
        if root is None: return None

        self.head = None
        self.node_pred = None
        travel_inorder(root)
        self.node_pred.right = self.head
        self.head.left = self.node_pred
        
        return self.head
```
# [剑指 Offer 31. 栈的压入、弹出序列](https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof/)

输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

 

示例 1：
```
输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
```
示例 2：
```
输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
```

提示：
```
0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed 是 popped 的排列。
```

代码：

```python3
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = []
        i = 0
        for pin in pushed:
            stack.append(pin)
            while stack and stack[-1] == popped[i]:
                stack.pop()
                i += 1
           
        return len(stack)==0
```
# [剑指 Offer 38. 字符串的排列](https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/)

输入一个字符串，打印出该字符串中字符的所有排列。

 

你可以以任意顺序返回这个字符串数组，但里面不能有重复元素。

 

示例:
```
输入：s = "abc"
输出：["abc","acb","bac","bca","cab","cba"]
```

限制：
```
1 <= s 的长度 <= 8
```

代码：

```python3
class Solution:
    def permutation(self, s: str) -> List[str]:
        s = list(s)
        def dfs(k: int) -> None:
            """k is the length of fixed part."""
            
            if k==len(s):
                # function `join`: ["a", "b", "c"] -> "abc"
                self.res.add( "".join(s) )
                return
            
            tried = set()
            for i in range(k, len(s)):
                if s[i] in tried:
                    continue
                tried.add(s[i])
                
                # back_track: 1. do sth, 2. dfs, 3. un-do sth
                s[k], s[i] = s[i], s[k]
                dfs(k+1)
                s[k], s[i] = s[i], s[k]
        
        self.res = set()
        dfs(0)
        return list(self.res)
```
# [剑指 Offer 32 - I. 从上到下打印二叉树](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-lcof/)

从上到下打印出二叉树的每个节点，同一层的节点按照从左到右的顺序打印。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

```
    3
   / \
  9  20
    /  \
   15   7
```
返回：

```
[3,9,20,15,7]
```

提示：

```
节点总数 <= 1000
```
代码：

```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[int]:
        
        if root is None: return []
        self.res = []
        q = deque([root])
        while q:
            qlen = len(q)
            for _ in range(qlen):
                node = q.popleft()
                self.res.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
                    
        return self.res
```

# [剑指 Offer 44. 数字序列中某一位的数字](https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/)

数字以0123456789101112131415…的格式序列化到一个字符序列中。在这个序列中，第5位（从下标0开始计数）是5，第13位是1，第19位是4，等等。

请写一个函数，求任意第n位对应的数字。

 

示例 1：

```
输入：n = 3
输出：3
```

示例 2：

```
输入：n = 11
输出：0
```


限制：
```
0 <= n < 2^31
```

代码：

```python3
class Solution:
    def findNthDigit(self, n: int) -> int:
        start, digit, count = 1, 1, 9
        while n > count:
            n -= count
            digit += 1
            start *= 10
            count = 9 * start * digit
        
        num = start + (n - 1) // digit

        return int(str(num)[(n - 1) % digit])
```
# [剑指 Offer 32 - III. 从上到下打印二叉树 III](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof/)

请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],
```
    3
   / \
  9  20
    /  \
   15   7
```
返回其层次遍历结果：

```
[
  [3],
  [20,9],
  [15,7]
]
```


提示：

```
节点总数 <= 1000
```

代码：

```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None: return []

        q = deque([root])
        level = 0
        ans = []
        while q:
            qlen = len(q)
            # get val from q
            if level%2==1:
                # if odd
                ans.append([q[i].val for i in range(qlen-1, -1, -1)])
            else:
                # if even
                ans.append([q[i].val for i in range(qlen)])
            
            # put val to q
            for _ in range(qlen):
                node = q.popleft()
                if node.left: 
                    q.append(node.left)
                if node.right: 
                    q.append(node.right)
                
            level += 1
            
        return ans
```
# [剑指 Offer 33. 二叉搜索树的后序遍历序列](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof/)

输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：
```
     5
    / \
   2   6
  / \
 1   3
```
示例 1：
```
输入: [1,6,3,2,5]
输出: false
```
示例 2：
```
输入: [1,3,2,6,5]
输出: true
```

提示：
```
数组长度 <= 1000
```

代码：

```python3
class Solution:
    def verifyPostorder(self, postorder: List[int]) -> bool:
        stack = []
        root = +inf

        for val in postorder[::-1]:
            if val > root:
                return False
            
            while stack and stack[-1]>val:
                root = stack.pop()
            stack.append(val)
            
        return True
```
# [剑指 Offer 34. 二叉树中和为某一值的路径](https://leetcode-cn.com/problems/er-cha-shu-zhong-he-wei-mou-yi-zhi-de-lu-jing-lcof/)

输入一棵二叉树和一个整数，打印出二叉树中节点值的和为输入整数的所有路径。从树的根节点开始往下一直到叶节点所经过的节点形成一条路径。

 

示例:
给定如下二叉树，以及目标和 target = 22，
```
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1
```
返回:
```
[
   [5,4,11,2],
   [5,8,4,5]
]
```

提示：
```
节点总数 <= 10000

```
代码：

```python3
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, target: int) -> List[List[int]]:
        def dfs(root: TreeNode, target: int, track: List[int]) -> None:
            is_leaf = root.left is None and root.right is None
            if is_leaf and target==0:
                self.res.append(list(track))
            
            for node in [root.left, root.right]:
                if node:
                    track.append(node.val)
                    dfs(node, target-node.val, track)
                    track.pop()
        
        self.res = []
        if root:
            dfs(root, target-root.val, [root.val])
        return self.res
```

# [剑指 Offer 56 - I. 数组中数字出现的次数](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-lcof/)

一个整型数组 nums 里除两个数字之外，其他数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

 

示例 1：
```
输入：nums = [4,1,4,6]
输出：[1,6] 或 [6,1]
```
示例 2：
```
输入：nums = [1,2,10,4,1,4,3,3]
输出：[2,10] 或 [10,2]
```

限制：
```
2 <= nums.length <= 10000
```
代码：

```python3
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:

        # xor over nums
        xor = 0
        for n in nums:
            xor ^= n

        # find mark
        mark = 1
        while True:
            if mark&xor>0:
                break
            mark <<= 1

        # split nums into two groups using mark
        # xor in each group
        d1, d2 = 0, 0
        for n in nums:
            if n&mark>0:
                d1 ^= n
            else:
                d2 ^= n
        
        return [d1, d2]
```

# [剑指 Offer 56 - II. 数组中数字出现的次数 II](https://leetcode-cn.com/problems/shu-zu-zhong-shu-zi-chu-xian-de-ci-shu-ii-lcof/)

在一个数组 nums 中除一个数字只出现一次之外，其他数字都出现了三次。请找出那个只出现一次的数字。

 

示例 1：
```
输入：nums = [3,4,3,3]
输出：4
```
示例 2：
```
输入：nums = [9,1,7,9,7,9,7]
输出：1
```

限制：
```
1 <= nums.length <= 10000
1 <= nums[i] < 2^31
```
代码：

```python3
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        hi, lo = 0, 0
        for i in nums:
            lo = (lo ^ i) & ~hi
            hi = (hi ^ i) & ~lo
        return lo
```

# [剑指 Offer 45. 把数组排成最小的数](https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/)

输入一个非负整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。

 

示例 1:
```
输入: [10,2]
输出: "102"
```
示例 2:
```
输入: [3,30,34,5,9]
输出: "3033459"
```

提示:
```
0 < nums.length <= 100
```
说明:

输出结果可能非常大，所以你需要返回一个字符串而不是整数
拼接起来的数字可能会有前导 0，最后结果不需要去掉前导 0

代码：

```python3
class Solution:
    def minNumber(self, nums: List[int]) -> str:
        key = cmp_to_key(lambda x, y: 1 if x+y>y+x else -1)
        return "".join(sorted([str(n) for n in nums], key=key))
```

# [剑指 Offer 46. 把数字翻译成字符串](https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/)

给定一个数字，我们按照如下规则把它翻译为字符串：0 翻译成 “a” ，1 翻译成 “b”，……，11 翻译成 “l”，……，25 翻译成 “z”。一个数字可能有多个翻译。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。

 

示例 1:
```
输入: 12258
输出: 5
解释: 12258有5种不同的翻译，分别是"bccfi", "bwfi", "bczi", "mcfi"和"mzi"
```

提示：
$$0 <= num < 2^{31}$$


代码：

```python3
class Solution:
    def translateNum(self, num: int) -> int:
        
        nums = str(num)
        prev_prev, prev, curr = 1, 1, 1
        for i in range(1, len(nums)):
            if nums[i-1:i+1]>"25" or nums[i-1:i+1]<"10":
                curr = prev
            else:
                curr = prev + prev_prev
            prev_prev, prev = prev, curr
        return curr
```

# [剑指 Offer 47. 礼物的最大价值](https://leetcode-cn.com/problems/li-wu-de-zui-da-jie-zhi-lcof/)

在一个 m*n 的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于 0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格、直到到达棋盘的右下角。给定一个棋盘及其上面的礼物的价值，请计算你最多能拿到多少价值的礼物？

 

示例 1:
```
输入: 
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 12
解释: 路径 1→3→5→2→1 可以拿到最多价值的礼物
```

提示：
```
0 < grid.length <= 200
0 < grid[0].length <= 200
```

代码：

```python3
class Solution:
    def maxValue(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        
        # first row
        for j in range(1, n):
            grid[0][j] += grid[0][j-1]
        # first colum
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        # other rows columns
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += max(grid[i-1][j], grid[i][j-1])

        return grid[m-1][n-1]
```

# [剑指 Offer 48. 最长不含重复字符的子字符串](https://leetcode-cn.com/problems/zui-chang-bu-han-zhong-fu-zi-fu-de-zi-zi-fu-chuan-lcof/)

请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。

 

示例 1:
```
输入: "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
```

提示：

```
s.length <= 40000
```
代码：

```python3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 方法一：
        len_max = 0
        len_prev = 0
        dic = {}
        for i in range(len(s)):
            c = s[i]
            len_curr = len_prev + 1
            if c in dic:
                len_curr = min(i - dic[c], len_curr)
            len_prev = len_curr
            len_max = max(len_max, len_curr)
            dic[c] = i
        return len_max

        # 方法二：
        i = -1
        res = 0
        dic = {}
        for j in range(len(s)):
            c = s[j]
            if c in dic:
                i = max(dic[c], i)
            dic[c] = j
            res = max(res, j - i)
        return res
```

# [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

 

示例:
```
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
```
说明:  
```
1 是丑数。
n 不超过1690。
```

代码：

```python3
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        dp = [1] * n
        i2, i3, i5 = 0, 0, 0
        for i in range(1, n):
            v2, v3, v5 = dp[i2] * 2, dp[i3] * 3, dp[i5] * 5
            dp[i] = min(v2, v3, v5)
            if dp[i] == v2:
                i2 += 1
            if dp[i] == v3:
                i3 += 1
            if dp[i] == v5:
                i5 += 1
        return dp[-1]
```

# [剑指 Offer 59 - II. 队列的最大值](https://leetcode-cn.com/problems/dui-lie-de-zui-da-zhi-lcof/)

请定义一个队列并实现函数 max_value 得到队列里的最大值，要求函数max_value、push_back 和 pop_front 的均摊时间复杂度都是O(1)。

若队列为空，pop_front 和 max_value 需要返回 -1

示例 1：
```
输入: 
["MaxQueue","push_back","push_back","max_value","pop_front","max_value"]
[[],[1],[2],[],[],[]]
输出: [null,null,null,2,1,2]
```
示例 2：
```
输入: 
["MaxQueue","pop_front","max_value"]
[[],[],[]]
输出: [null,-1,-1]
```

限制：
```
1 <= push_back,pop_front,max_value的总操作数 <= 10000
1 <= value <= 10^5
```
代码：

```python3
class MaxQueue:

    def __init__(self):
        self.q = deque()
        self.max_q = deque()

    def max_value(self) -> int:
        if self.max_q:
            return self.max_q[0]
        else:
            return -1

    def push_back(self, value: int) -> None:
        self.q.append(value)
        while self.max_q and value > self.max_q[-1]:
            self.max_q.pop()
        self.max_q.append(value)


    def pop_front(self) -> int:
        if len(self.max_q) == 0:
            return -1
        value = self.q.popleft()
        if value == self.max_q[0]:
            self.max_q.popleft()
        return value
```


# [剑指 Offer 66. 构建乘积数组](https://leetcode-cn.com/problems/gou-jian-cheng-ji-shu-zu-lcof/)

给定一个数组 A[0,1,…,n-1]，请构建一个数组 B[0,1,…,n-1]，其中 B[i] 的值是数组 A 中除了下标 i 以外的元素的积, 即 B[i]=A[0]×A[1]×…×A[i-1]×A[i+1]×…×A[n-1]。不能使用除法。


示例:
```
输入: [1,2,3,4,5]
输出: [120,60,40,30,24]
```

提示：
```
所有元素乘积之和不会溢出 32 位整数
a.length <= 100000
```
代码：

```python3
class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        n = len(a)
        if n == 0:
            return []
        res = [1] * n
        for i in range(1, n):
            res[i] = res[i - 1] * a[i - 1]
        tmp = 1
        for i in range(n - 2, -1, -1):
            tmp *= a[i + 1]
            res[i] *= tmp
        return res
```


# [剑指 Offer 60. n个骰子的点数](https://leetcode-cn.com/problems/nge-tou-zi-de-dian-shu-lcof/)

把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。
 

你需要用一个浮点数数组返回答案，其中第 i 个元素代表这 n 个骰子所能掷出的点数集合中第 i 小的那个的概率。
 

示例 1:
```
输入: 1
输出: [0.16667,0.16667,0.16667,0.16667,0.16667,0.16667]
```

示例 2:
```
输入: 2
输出: [0.02778,0.05556,0.08333,0.11111,0.13889,0.16667,0.13889,0.11111,0.08333,0.05556,0.02778]
```

限制：
```
1 <= n <= 11
```
代码：

```python3
class Solution:
    def dicesProbability(self, n: int) -> List[float]:
        prev = [1/6] * 6
        for i in range(2, n+1):
            curr = [0] * (5 * i + 1)
            for j in range(len(prev)):
                for k in range(6):
                    curr[j + k] += prev[j] / 6
            prev = curr
        return prev
```


# [剑指 Offer 67. 把字符串转换成整数](https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof/)

写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似的库函数。

 

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

说明：
```
假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
```
示例 1:
```
输入: "42"
输出: 42
```
示例 2:
```
输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
```
示例 3:
```
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
```

示例 4:
```
输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。
```
示例 5:
```
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。 
     因此返回 INT_MIN (−231) 。
```

代码：

```python3
class Solution:
    def strToInt(self, str: str) -> int:
        s = len(str)
        if s == 0:
            return 0
        i = 0
        while i < s and str[i] == " ":
            i += 1
        if i == s:
            return 0
        h = str[i]
        sign = 1
        num = 0
        is_digit = lambda c: "0" <= c <= "9"
        int_min = -2147483648
        int_max = 2147483647
        if h == "-":
            sign = -1
        elif h == "+":
            pass
        elif is_digit(h):
            num = int(h)
        else:
            return 0
        for c in str[i+1:]:
            if not is_digit(c):
                return sign * num
            num = num * 10 + int(c)
            if sign == 1 and num >= int_max:
                return int_max
            if sign == -1 and num >= int_max + 1:
                return int_min
        return sign * num
```


# [剑指 Offer 63. 股票的最大利润](https://leetcode-cn.com/problems/gu-piao-de-zui-da-li-run-lcof/)

假设把某股票的价格按照时间先后顺序存储在数组中，请问买卖该股票一次可能获得的最大利润是多少？

 

示例 1:
```
输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。
```
示例 2:
```
输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。
```

限制：
```
0 <= 数组长度 <= 10^5
```
代码：

```python3
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        price_min = +inf
        profit_max = 0
        for p in prices:
            profit_max = max(profit_max, p - price_min)
            if p < price_min:
                price_min = p
        return profit_max
```