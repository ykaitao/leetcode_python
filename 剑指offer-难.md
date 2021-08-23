# [剑指 Offer 19. 正则表达式匹配](https://leetcode-cn.com/problems/zheng-ze-biao-da-shi-pi-pei-lcof/)

请实现一个函数用来匹配包含'. '和'\*'的正则表达式。模式中的字符'.'表示任意一个字符，而'\*'表示它前面的字符可以出现任意次（含0次）。在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但与"aa.a"和"ab*a"均不匹配。

示例 1:
```
输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
```

示例 2:
```
输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
```

示例 3:
```
输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
```

示例 4:
```
输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
```

示例 5:
```
输入:
s = "mississippi"
p = "mis*is*p*."
输出: false
```

```
s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母以及字符 . 和 *，无连续的 '*'。
```

代码：

```python3
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 方法一，动态规划:
        m, n = len(s), len(p)

        # dp[i][j]: is_match from end tail (i, j)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        for i in range(m, -1, -1):
            for j in range(n, -1, -1):
                if j == n:
                    dp[i][j] = (i == m)
                else:
                    first_match = (i < m) and (s[i] == p[j] or p[j] == ".")
                    if (j + 1) < n and p[j + 1] == "*":
                        dp[i][j] = (first_match and dp[i + 1][j]) or dp[i][j + 2]
                    else:
                        dp[i][j] = first_match and dp[i + 1][j + 1]
        return dp[0][0]


class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 方法二：
        def dfs(i: int, j: int) -> bool:
            if (i, j) in self.memo:
                return self.memo[(i, j)]
            if j >= len(p):
                res = (i >= len(s))
            else:
                first_match = (i < len(s) and (s[i] == p[j] or p[j] == "."))
                if (j + 1) <= (len(p) - 1) and p[j + 1] == "*":
                    res = (
                        first_match and dfs(i + 1, j) or
                        dfs(i, j + 2)
                    )
                else:
                    res = (
                        first_match and dfs(i + 1, j + 1)
                    )
            self.memo[(i, j)] = res
            return res

        self.memo = {}
        return dfs(0, 0)
```

# [剑指 Offer 41. 数据流中的中位数](https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof/)

如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，
```
[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5
```

设计一个支持以下两种操作的数据结构：
```
void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
```

示例 1：
```
输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
```

示例 2：
```
输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]
```

限制：
```
最多会对 addNum、findMedian 进行 50000 次调用。
```

代码：

```python3
from heapq import *  # Python 中 heapq 模块是小顶堆。实现 大顶堆 方法： 小顶堆的插入和弹出操作均将元素 取反 即可。
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
    def __init__(self):
        self.A = [] # 小顶堆，保存较大的一半
        self.B = [] # 大顶堆，保存较小的一半

    def addNum(self, num: int) -> None:
        if len(self.A) != len(self.B):
            heappush(self.A, num)
            heappush(self.B, -heappop(self.A))
        else:
            heappush(self.B, -num)
            heappush(self.A, -heappop(self.B))

    def findMedian(self) -> float:
        return self.A[0] if len(self.A) != len(self.B) else (self.A[0] - self.B[0]) / 2.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

# [剑指 Offer 37. 序列化二叉树](https://leetcode-cn.com/problems/xu-lie-hua-er-cha-shu-lcof/)

请实现两个函数，分别用来序列化和反序列化二叉树。

你需要设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

```
提示：输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。
```
 

示例：

<img src=https://assets.leetcode.com/uploads/2020/09/15/serdeser.jpg width="700"/>

```
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]
```

代码：

```python3
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return "$,"

        return (
            f"{root.val}," +
            self.serialize(root.left) +
            self.serialize(root.right)
        )
        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def build_tree() ->TreeNode:
            val = data_list[self.i]
            self.i += 1
            if val != "$":
                root = TreeNode(val)
                root.left = build_tree()
                root.right = build_tree()
                return root
            else:
                return 

        data_list = data.split(",")[:-1]
        if data_list[0] == "$":
            return 
        self.i = 0
        return build_tree()
```

# [剑指 Offer 43. 1～n 整数中 1 出现的次数](https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof/)

输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

 

示例 1：
```
输入：n = 12
输出：5
```

示例 2：
```
输入：n = 13
输出：6
```

限制：
```
1 <= n < 2^31
```

代码：

```python3
class Solution:
    def countDigitOne(self, n: int) -> int:
        nums = str(n)
        N = len(nums)
        
        high = 0
        low = n
        digit = 10 ** (N - 1)
        count = 0
        for i in range(N):
            d = int(nums[i])
            low = low % digit
            if d == 0:
                count += high * digit 
            elif d == 1:
                count += (low + 1) + high * digit
            elif d > 1:
                count += (high + 1) * digit
            high = high * 10 + d
            digit = digit // 10
        return count
```

# [剑指 Offer 51. 数组中的逆序对](https://leetcode-cn.com/problems/shu-zu-zhong-de-ni-xu-dui-lcof/)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数。

 

示例 1:
```
输入: [7,5,6,4]
输出: 5
```

限制：
```
0 <= 数组长度 <= 50000
```

代码：

```python3
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def get_reserse_pair(left: int, right: int) -> int:
            if left >= right:
                return 0
            
            mid = left + (right - left) // 2

            count_left = get_reserse_pair(left, mid)
            count_right = get_reserse_pair(mid + 1, right)
            if nums[mid] <= nums[mid + 1]:
                return count_left + count_right

            count_merge = merge_and_count(left, mid, right)
            return count_left + count_right + count_merge
        
        def merge_and_count(left: int, mid: int, right: int) -> int:
            for i in range(left, right + 1):
                tmp[i] = nums[i]
            i = left
            j = mid + 1
            count = 0
            for k in range(left, right + 1):
                if i > mid:
                    nums[k] = tmp[j]
                    j += 1
                elif j > right:
                    nums[k] = tmp[i]
                    i += 1
                elif tmp[i] <= tmp[j]:
                    nums[k] = tmp[i]
                    i += 1
                elif tmp[i] > tmp[j]:
                    nums[k] = tmp[j]
                    j += 1
                    count += (mid - i + 1)
            return count
        
        tmp = list(nums)
        return get_reserse_pair(0, len(nums) - 1)
```

# [剑指 Offer 59 - I. 滑动窗口的最大值](https://leetcode-cn.com/problems/hua-dong-chuang-kou-de-zui-da-zhi-lcof/)

给定一个数组 nums 和滑动窗口的大小 k，请找出所有滑动窗口里的最大值。

示例:
```
输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
输出: [3,3,5,5,6,7] 
解释: 

  滑动窗口的位置                最大值
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```

提示：
```
你可以假设 k 总是有效的，在输入数组不为空的情况下，1 ≤ k ≤ 输入数组的大小。
```

代码：

```python3
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        stack = []
        res = []
        for i, num in enumerate(nums):
            while stack and nums[stack[-1]] < num:
                stack.pop()

            while stack and stack[0] <= i - k:
                stack.pop(0)

            stack.append(i)
            if i >= (k - 1):
                res.append(nums[stack[0]])
        return res
```