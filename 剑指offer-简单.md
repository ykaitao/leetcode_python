# [剑指 Offer 09. 用两个栈实现队列](https://leetcode-cn.com/problems/yong-liang-ge-zhan-shi-xian-dui-lie-lcof/)

用两个栈实现一个队列。队列的声明如下，请实现它的两个函数 appendTail 和 deleteHead ，分别完成在队列尾部插入整数和在队列头部删除整数的功能。(若队列中没有元素，deleteHead 操作返回 -1 )

 

示例 1：
```
输入：
["CQueue","appendTail","deleteHead","deleteHead"]
[[],[3],[],[]]
输出：[null,null,3,-1]
```
示例 2：
```
输入：
["CQueue","deleteHead","appendTail","appendTail","deleteHead","deleteHead"]
[[],[],[5],[2],[],[]]
输出：[null,-1,null,null,5,2]
```
提示：
```
1 <= values <= 10000
最多会对 appendTail、deleteHead 进行 10000 次调用
```

代码：
```python3
class CQueue:

    def __init__(self):
        self.stack_A = []  # 入栈
        self.stack_B = []  # 出栈

    def appendTail(self, value: int) -> None:
        self.stack_A.append(value)

    def deleteHead(self) -> int:
        if self.stack_B:
            return self.stack_B.pop()
        if not self.stack_A:
            return -1
        while self.stack_A:
            self.stack_B.append(self.stack_A.pop())
        return self.stack_B.pop()
```

# [剑指 Offer 10- I. 斐波那契数列](https://leetcode-cn.com/problems/fei-bo-na-qi-shu-lie-lcof/)

写一个函数，输入 n ，求斐波那契（Fibonacci）数列的第 n 项（即 F(N)）。斐波那契数列的定义如下：

F(0) = 0,   F(1) = 1
F(N) = F(N - 1) + F(N - 2), 其中 N > 1.
斐波那契数列由 0 和 1 开始，之后的斐波那契数就是由之前的两数相加而得出。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。



示例 1：
```
输入：n = 2
输出：1
```
示例 2：
```
输入：n = 5
输出：5
```

提示：
```
0 <= n <= 100
```

代码：
```python3
class Solution:
    def fib(self, n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

# [剑指 Offer 03. 数组中重复的数字](https://leetcode-cn.com/problems/shu-zu-zhong-zhong-fu-de-shu-zi-lcof/)

找出数组中重复的数字。


在一个长度为 n 的数组 nums 里的所有数字都在 0～n-1 的范围内。数组中某些数字是重复的，但不知道有几个数字重复了，也不知道每个数字重复了几次。请找出数组中任意一个重复的数字。

示例 1：
```
输入：
[2, 3, 1, 0, 2, 5, 3]
输出：2 或 3 
```

限制：
```
2 <= n <= 100000
```

代码：
```python3
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
            
        for i in range(len(nums)):
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
            if i != nums[i] and nums[i] == nums[nums[i]]:
                return nums[i]
        return None
```

# [剑指 Offer 10- II. 青蛙跳台阶问题](https://leetcode-cn.com/problems/qing-wa-tiao-tai-jie-wen-ti-lcof/)

一只青蛙一次可以跳上1级台阶，也可以跳上2级台阶。求该青蛙跳上一个 n 级的台阶总共有多少种跳法。

答案需要取模 1e9+7（1000000007），如计算初始结果为：1000000008，请返回 1。

示例 1：
```
输入：n = 2
输出：2
```
示例 2：
```
输入：n = 7
输出：21
```
示例 3：
```
输入：n = 0
输出：1
```
提示：
```
0 <= n <= 100
```

代码：
```python3
class Solution:
    def numWays(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n):
            a, b = b, a + b
        return a % 1000000007
```

# [剑指 Offer 11. 旋转数组的最小数字](https://leetcode-cn.com/problems/xuan-zhuan-shu-zu-de-zui-xiao-shu-zi-lcof/)

把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个递增排序的数组的一个旋转，输出旋转数组的最小元素。例如，数组 [3,4,5,1,2] 为 [1,2,3,4,5] 的一个旋转，该数组的最小值为1。  

示例 1：
```
输入：[3,4,5,1,2]
输出：1
```
示例 2：
```
输入：[2,2,2,0,1]
输出：0
```

代码：
```python3
class Solution:
    def minArray(self, numbers: List[int]) -> int:
        i, j = 0, len(numbers) - 1
        while i <= j:
            m = i + (j - i) // 2
            if numbers[m] > numbers[j]:
                i = m + 1
            elif numbers[m] < numbers[j]:
                j = m
            else:
                j -= 1
        return numbers[i]
```

# [剑指 Offer 05. 替换空格](https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof/)

请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

 

示例 1：
```
输入：s = "We are happy."
输出："We%20are%20happy."
```

限制：
```
0 <= s 的长度 <= 10000
```

代码：
```python3
class Solution:
    def replaceSpace(self, s: str) -> str:
        # s = s.split(" ")
        # return "%20".join(s)

        res = []
        for w in s:
            if w == " ":
                res.append("%20")
            else:
                res.append(w)
        return "".join(res)
```

# [剑指 Offer 06. 从尾到头打印链表](https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof/)

输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

 

示例 1：
```
输入：head = [1,3,2]
输出：[2,3,1]
```

限制：
```
0 <= 链表长度 <= 10000
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        # 翻转列表
        res = []
        while head:
            res.append(head.val)
            head = head.next
        return res[::-1]
```

# [剑指 Offer 25. 合并两个排序的链表](https://leetcode-cn.com/problems/he-bing-liang-ge-pai-xu-de-lian-biao-lcof/)

输入两个递增排序的链表，合并这两个链表并使新链表中的节点仍然是递增排序的。

示例1：
```
输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4
```
限制：
```
0 <= 链表长度 <= 1000
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        node = head = ListNode()
        while l1 and l2:
            if l1.val <= l2.val:
                node.next, l1 = ListNode(l1.val), l1.next
            else:
                node.next, l2 = ListNode(l2.val), l2.next
            node = node.next
        node.next = l1 if l1 else l2
        return head.next
```

# [剑指 Offer 27. 二叉树的镜像](https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/)

请完成一个函数，输入一个二叉树，该函数输出它的镜像。

例如输入：
```
     4
   /   \
  2     7
 / \   / \
1   3 6   9
```
镜像输出：
```
     4
   /   \
  7     2
 / \   / \
9   6 3   1
```
 

示例 1：
```
输入：root = [4,2,7,1,3,6,9]
输出：[4,7,2,9,6,3,1]
```

限制：
```
0 <= 节点个数 <= 1000
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
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        # 方法一，递归
        if root is None:
            return
        root.left, root.right = self.mirrorTree(root.right), self.mirrorTree(root.left)
        return root

class Solution:
    def mirrorTree(self, root: TreeNode) -> TreeNode:
        # 方法二，辅助栈
        if root is None: return
        
        stack = [root]
        while stack:
            node = stack.pop()
            if node.left: stack.append(node.left)
            if node.right: stack.append(node.right)
            node.left, node.right = node.right, node.left
        return root
```

# [剑指 Offer 28. 对称的二叉树](https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/)

请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
```
    1
   / \
  2   2
   \   \
   3    3
```
 

示例 1：
```
输入：root = [1,2,2,3,4,4,3]
输出：true
```
示例 2：
```
输入：root = [1,2,2,null,3,null,3]
输出：false
```

限制：
```
0 <= 节点个数 <= 1000
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
    def isSymmetric(self, root: TreeNode) -> bool:
        def is_mirror(left: TreeNode, right: TreeNode) -> bool:
            if left is None and right is None:
                return True
            if left is None or right is None or left.val != right.val:
                return False
            return is_mirror(left.left, right.right) and is_mirror(left.right, right.left)

        return is_mirror(root.left, root.right) if root else True
```

# [剑指 Offer 21. 调整数组顺序使奇数位于偶数前面](https://leetcode-cn.com/problems/diao-zheng-shu-zu-shun-xu-shi-qi-shu-wei-yu-ou-shu-qian-mian-lcof/)

输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有奇数位于数组的前半部分，所有偶数位于数组的后半部分。

 

示例：
```
输入：nums = [1,2,3,4]
输出：[1,3,2,4] 
注：[3,1,2,4] 也是正确的答案之一。
```

提示：
```
0 <= nums.length <= 50000
1 <= nums[i] <= 10000
```

代码：
```python3
class Solution:
    def exchange(self, nums: List[int]) -> List[int]:

        i, j = 0, len(nums) - 1
        while i < j:
            while i < j and nums[i] & 1 == 1:
                i += 1
            while i < j and nums[j] & 1 == 0:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        return nums
```

# [剑指 Offer 15. 二进制中1的个数](https://leetcode-cn.com/problems/er-jin-zhi-zhong-1de-ge-shu-lcof/)

编写一个函数，输入是一个无符号整数（以二进制串的形式），返回其二进制表达式中数字位数为 '1' 的个数（也被称为 汉明重量).）。

 

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用 二进制补码 记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
 

示例 1：
```
输入：n = 11 (控制台输入 00000000000000000000000000001011)
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
```
示例 2：
```
输入：n = 128 (控制台输入 00000000000000000000000010000000)
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
```
示例 3：
```
输入：n = 4294967293 (控制台输入 11111111111111111111111111111101，部分语言中 n = -3）
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
```

提示：
```
输入必须是长度为 32 的 二进制串 。
```

代码：
```python3
class Solution:
    def hammingWeight(self, n: int) -> int:
        # 解法一
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count

class Solution:
    def hammingWeight(self, n: int) -> int:
        # 解法二：
        c = 0
        while n > 0:
            n = n & (n - 1)
            c += 1
        return c
```

# [剑指 Offer 29. 顺时针打印矩阵](https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof/)

此题详情和解法请见[腾讯精选练习-54](https://github.com/ykaitao/leetcode_python/blob/master/%E8%85%BE%E8%AE%AF%E7%B2%BE%E9%80%89%E7%BB%83%E4%B9%A050%E9%A2%98-%E4%B8%AD%E7%AD%89.md#54-%E8%9E%BA%E6%97%8B%E7%9F%A9%E9%98%B5)

# [剑指 Offer 22. 链表中倒数第k个节点](https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof/)

输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链表的倒数第 3 个节点是值为 4 的节点。

 

示例：
```
给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        fast = slow = head
        for _ in range(k):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        return slow
```

# [剑指 Offer 17. 打印从1到最大的n位数](https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof/)

输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

示例 1:
```
输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
```

说明：
```
用返回一个整数列表来代替打印
n 为正整数
```

代码：
```python3
class Solution:
    def printNumbers(self, n: int) -> List[int]:
        return [x for x in range(1, 10 ** n)]
```

# [剑指 Offer 24. 反转链表](https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof/)

定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

 

示例:
```
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
```

限制：
```
0 <= 节点个数 <= 5000
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        node_prev = None
        node_curr = head

        while node_curr:
            node_next = node_curr.next
            node_curr.next = node_prev
            node_prev, node_curr = node_curr, node_next
            
        return node_prev
```

# [剑指 Offer 18. 删除链表的节点](https://leetcode-cn.com/problems/shan-chu-lian-biao-de-jie-dian-lcof/)

给定单向链表的头指针和一个要删除的节点的值，定义一个函数删除该节点。

返回删除后的链表的头节点。

注意：此题对比原题有改动

示例 1:
```
输入: head = [4,5,1,9], val = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
```
示例 2:
```
输入: head = [4,5,1,9], val = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
```

说明：
```
题目保证链表中节点的值互不相同
若使用 C 或 C++ 语言，你不需要 free 或 delete 被删除的节点
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        if head.val == val:
            return head.next
        curr = head.next
        prev = head_copy = head
        while curr and curr.val != val:
            prev, curr =curr, curr.next
        if curr:
            prev.next = curr.next
        return head_copy
        
```

# [剑指 Offer 40. 最小的k个数](https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof/)

输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

示例 1：
```
输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
```
示例 2：
```
输入：arr = [0,1,2,1], k = 1
输出：[0]
```

限制：
```
0 <= k <= arr.length <= 10000
0 <= arr[i] <= 10000
```

代码：
```python3
class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 解法一：
        heap_max = [-arr[i] for i in range(k)]
        heapify(heap_max)
        
        for num in arr[k:]:
            heappush(heap_max, -num)
            heappop(heap_max)

        return [-heappop(heap_max) for _ in range(k)]


class Solution:
    def getLeastNumbers(self, arr: List[int], k: int) -> List[int]:
        # 解法二：
        count_arr = [0] * 10001
        for n in arr:
            count_arr[n] += 1

        res = []
        for i, c in enumerate(count_arr):
            if len(res) >= k:
                break
            
            if c > 0:
                res.extend([i] * c)
        return res[:k]
```

# [剑指 Offer 30. 包含min函数的栈](https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof/)
此题可对比[剑指offer-59](https://github.com/ykaitao/leetcode_python/blob/master/%E5%89%91%E6%8C%87offer-%E4%B8%AD%E7%AD%89.md#%E5%89%91%E6%8C%87-offer-59---ii-%E9%98%9F%E5%88%97%E7%9A%84%E6%9C%80%E5%A4%A7%E5%80%BC)学习

定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

示例:
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
```

提示：
```
各函数的调用总次数不超过 20000 次
```

代码：
```python3
class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack_A = []
        self.stack_B = []

    def push(self, x: int) -> None:
        self.stack_A.append(x)
        if not self.stack_B or self.stack_B[-1] >= x:
            self.stack_B.append(x)

    def pop(self) -> None:
        if self.stack_A.pop() == self.stack_B[-1]:
            self.stack_B.pop()

    def top(self) -> int:
        return self.stack_A[-1]


    def min(self) -> int:
        return self.stack_B[-1]



# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.min()
```

# [剑指 Offer 42. 连续子数组的最大和](https://leetcode-cn.com/problems/lian-xu-zi-shu-zu-de-zui-da-he-lcof/)

输入一个整型数组，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

要求时间复杂度为O(n)。

 

示例1:
```
输入: nums = [-2,1,-3,4,-1,2,1,-5,4]
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
```

提示：
```
1 <= arr.length <= 10^5
-100 <= arr[i] <= 100
```

代码：
```python3
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            nums[i] += max(nums[i - 1], 0)
        return max(nums)


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 不改变原数组：
        ans = curr = nums[0]
        for v in nums[1:]:
            curr = v + max(0, curr)
            ans = max(ans, curr)
        return ans
```

# [剑指 Offer 39. 数组中出现次数超过一半的数字](https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof/)

数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

示例 1:
```
输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
```

限制：
```
1 <= 数组长度 <= 50000
```

代码：
```python3
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        for num in nums:
            if count == 0:
                x = num
            count += 1 if x == num else -1
        return x
```

# [剑指 Offer 32 - II. 从上到下打印二叉树 II](https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof/)

从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 

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
  [9,20],
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
        if root is None:
            return []
        
        q = deque()
        q.append(root)
        res = []
        while q:
            tmp = []
            n = len(q)
            for _ in range(n):
                node = q.popleft()
                if node:
                    tmp.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(tmp)
        return res
```

# [剑指 Offer 50. 第一个只出现一次的字符](https://leetcode-cn.com/problems/di-yi-ge-zhi-chu-xian-yi-ci-de-zi-fu-lcof/)

在字符串 s 中找出第一个只出现一次的字符。如果没有，返回一个单空格。 s 只包含小写字母。

示例:
```
s = "abaccdeff"
返回 "b"

s = "" 
返回 " "
```

限制：
```
0 <= s 的长度 <= 50000
```

代码：
```python3
class Solution:
    def firstUniqChar(self, s: str) -> str:
        # 方法一：
        tmp = collections.Counter(s)
        for c in s:
            if tmp[c] == 1:
                return c
        return " "


class Solution:
    def firstUniqChar(self, s: str) -> str:
        # 方法二：
        dic = OrderedDict()
        for c in s:
            dic[c] = 1 + (dic[c] if c in dic else 0)
        
        for k, v in dic.items():
            if v == 1:
                return k
        return " "
```

# [剑指 Offer 55 - I. 二叉树的深度](https://leetcode-cn.com/problems/er-cha-shu-de-shen-du-lcof/)

输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。

例如：

给定二叉树 [3,9,20,null,null,15,7]，
```
    3
   / \
  9  20
    /  \
   15   7
```
返回它的最大深度 3 。

 

提示：
```
节点总数 <= 10000
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
    def maxDepth(self, root: TreeNode) -> int:
        # 深度优先：
        if root is None:
            return 0
        return max(
            self.maxDepth(root.left),
            self.maxDepth(root.right)
        ) + 1



class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 广度优先：  
        if root is None:
            return 0      
        count = 0
        q = deque()
        q.append(root)
        while q:
            n = len(q)
            for i in range(n):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            count += 1
        return count
```

# [剑指 Offer 57. 和为s的两个数字](https://leetcode-cn.com/problems/he-wei-sde-liang-ge-shu-zi-lcof/)

输入一个递增排序的数组和一个数字s，在数组中查找两个数，使得它们的和正好是s。如果有多对数字的和等于s，则输出任意一对即可。

 

示例 1：
```
输入：nums = [2,7,11,15], target = 9
输出：[2,7] 或者 [7,2]
```
示例 2：
```
输入：nums = [10,26,30,31,47,60], target = 40
输出：[10,30] 或者 [30,10]
```

限制：
```
1 <= nums.length <= 10^5
1 <= nums[i] <= 10^6
```

代码：
```python3
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        i, j = 0, len(nums) - 1
        while i < j:
            sumij = nums[i] + nums[j]
            if sumij == target:
                return [nums[i], nums[j]]
            elif sumij < target:
                i += 1
            else:
                j -= 1
        return []
```

# [剑指 Offer 57 - II. 和为s的连续正数序列](https://leetcode-cn.com/problems/he-wei-sde-lian-xu-zheng-shu-xu-lie-lcof/)

输入一个正整数 target ，输出所有和为 target 的连续正整数序列（至少含有两个数）。

序列内的数字由小到大排列，不同序列按照首个数字从小到大排列。

 

示例 1：
```
输入：target = 9
输出：[[2,3,4],[4,5]]
```
示例 2：
```
输入：target = 15
输出：[[1,2,3,4,5],[4,5,6],[7,8]]
```

限制：
```
1 <= target <= 10^5
```

代码：
```python3
class Solution:
    def findContinuousSequence(self, target: int) -> List[List[int]]:
        i, j = 1, 1
        sumi_j = 0
        res = []
        start_max = target // 2
        while i <= start_max:
            if sumi_j > target:
                sumi_j -= i
                i += 1
            elif sumi_j < target:
                sumi_j += j
                j += 1
            else:
                res.append([x for x in range(i, j)])
                sumi_j = sumi_j - i
                i += 1
        return res
```

# [剑指 Offer 52. 两个链表的第一个公共节点](https://leetcode-cn.com/problems/liang-ge-lian-biao-de-di-yi-ge-gong-gong-jie-dian-lcof/)

输入两个链表，找出它们的第一个公共节点。

如下面的两个链表：
<img src=https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_statement.png width="700">


在节点 c1 开始相交。

 

示例 1：

<img src=https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_1.png width="700">
```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：Reference of the node with value = 8
输入解释：相交节点的值为 8 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
```

示例 2：

<img src=https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_2.png width="700">
```
输入：intersectVal = 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
输出：Reference of the node with value = 2
输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
```

示例 3：

<img src=https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/160_example_3.png width="700">
```
输入：intersectVal = 0, listA = [2,6,4], listB = [1,5], skipA = 3, skipB = 2
输出：null
输入解释：从各自的表头开始算起，链表 A 为 [2,6,4]，链表 B 为 [1,5]。由于这两个链表不相交，所以 intersectVal 必须为 0，而 skipA 和 skipB 可以是任意值。
解释：这两个链表不相交，因此返回 null。
```

注意：
```
如果两个链表没有交点，返回 null.
在返回结果后，两个链表仍须保持原有的结构。
可假定整个链表结构中没有循环。
程序尽量满足 O(n) 时间复杂度，且仅用 O(1) 内存。
```

代码：
```python3
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        nodeA = headA
        nodeB = headB

        while nodeA != nodeB:
                nodeA = nodeA.next if nodeA else headB
                nodeB = nodeB.next if nodeB else headA
        return nodeA
```

# [剑指 Offer 58 - I. 翻转单词顺序](https://leetcode-cn.com/problems/fan-zhuan-dan-ci-shun-xu-lcof/)

输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student. "，则输出"student. a am I"。

 

示例 1：
```
输入: "the sky is blue"
输出: "blue is sky the"
```
示例 2：
```
输入: "  hello world!  "
输出: "world! hello"
解释: 输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
```
示例 3：
```
输入: "a good   example"
输出: "example good a"
解释: 如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

说明：
```
无空格字符构成一个单词。
输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
```

代码：
```python3
class Solution:
    def reverseWords(self, s: str) -> str:
        s = s.strip(" ")
        n = len(s)
        res = []
        i = j = n - 1
        while i >= 0:
            while i >= 0 and s[i] != " ": i -= 1
            res.append(s[i + 1:j + 1])
            while i >= 0 and s[i] == " ": i -= 1
            j = i
        return " ".join(res)


class Solution:
    def reverseWords(self, s: str) -> str:
        # 解法二：
        return " ".join(s.split()[::-1])
```

# [剑指 Offer 53 - I. 在排序数组中查找数字 I](https://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/)

统计一个数字在排序数组中出现的次数。

 

示例 1:
```
输入: nums = [5,7,7,8,8,10], target = 8
输出: 2
```
示例 2:
```
输入: nums = [5,7,7,8,8,10], target = 6
输出: 0
```

提示：
```
0 <= nums.length <= 105
-109 <= nums[i] <= 109
nums 是一个非递减数组
-109 <= target <= 109
```

代码：
```python3
class Solution:
    def search(self, nums: List[int], target: int) -> int:

        def searchSide(side: str) -> int:
            left, right = 0, n - 1
            while left <= right:
                m = left + (right - left) // 2
                if nums[m] < target:
                    left = m + 1
                elif nums[m] > target:
                    right = m - 1
                else:
                    if side == "left":
                        right = m - 1
                    else:
                        left = m + 1
            return left if side == "left" else right

        n = len(nums)
        left = searchSide("left")
        if left >= n or nums[left] != target:
            return 0
        else:
            right = searchSide("right")
            
        return right - left + 1
```

# [剑指 Offer 58 - II. 左旋转字符串](https://leetcode-cn.com/problems/zuo-xuan-zhuan-zi-fu-chuan-lcof/)

字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

 

示例 1：
```
输入: s = "abcdefg", k = 2
输出: "cdefgab"
```
示例 2：
```
输入: s = "lrloseumgh", k = 6
输出: "umghlrlose"
```

限制：
```
1 <= k < s.length <= 10000
```

代码：
```python3
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        # return s[n:] + s[:n]

        res = ""
        for i in range(n, n + len(s)):
            res += s[i % len(s)]
        return res
```

# [剑指 Offer 53 - II. 0～n-1中缺失的数字](https://leetcode-cn.com/problems/que-shi-de-shu-zi-lcof/)

一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0～n-1之内。在范围0～n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字。

 

示例 1:
```
输入: [0,1,3]
输出: 2
```
示例 2:
```
输入: [0,1,2,3,4,5,6,7,9]
输出: 8
```

限制：
```
1 <= 数组长度 <= 10000
```

代码：
```python3
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        while i <= j:
            m = i + (j - i) // 2
            if nums[m] == m:
                i = m + 1
            else:
                j = m - 1
        return i
```

# [剑指 Offer 54. 二叉搜索树的第k大节点](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-di-kda-jie-dian-lcof/)

给定一棵二叉搜索树，请找出其中第k大的节点。

 

示例 1:
```
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 4
```
示例 2:
```
输入: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
输出: 4
```

限制：
```
1 ≤ k ≤ 二叉搜索树元素个数
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
    def kthLargest(self, root: TreeNode, k: int) -> int:
        def dfs(root: TreeNode) -> None:
            if root is None:
                return None
            dfs(root.right)
            if self.k == 0:
                return
            self.k -= 1
            if self.k == 0:
                self.res = root.val
            dfs(root.left)

        self.k = k
        self.res = None
        dfs(root)
        return self.res
```

# [剑指 Offer 65. 不用加减乘除做加法](https://leetcode-cn.com/problems/bu-yong-jia-jian-cheng-chu-zuo-jia-fa-lcof/)

写一个函数，求两个整数之和，要求在函数体内不得使用 “+”、“-”、“*”、“/” 四则运算符号。

 

示例:
```
输入: a = 1, b = 1
输出: 2
```

提示：
```
a, b 均可能是负数或 0
结果不会溢出 32 位整数
```

代码：
```python3
class Solution:
    def add(self, a: int, b: int) -> int:
        x = 0xffffffff
        a, b = a & x, b & x
        while b != 0:
            a, b = (a ^ b), (a & b) << 1 & x
        return a if a <= 0x7fffffff else ~(a & x)
```

# [剑指 Offer 61. 扑克牌中的顺子](https://leetcode-cn.com/problems/bu-ke-pai-zhong-de-shun-zi-lcof/)

从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2～10为数字本身，A为1，J为11，Q为12，K为13，而大、小王为 0 ，可以看成任意数字。A 不能视为 14。

 

示例 1:
```
输入: [1,2,3,4,5]
输出: True
```

示例 2:
```
输入: [0,0,1,2,5]
输出: True
```

限制：
```
数组长度为 5 

数组的数取值为 [0, 13] .
```

代码：
```python3
class Solution:
    def isStraight(self, nums: List[int]) -> bool:
        k = 0
        nums.sort()
        for i in range(4):
            if nums[i] == 0:
                k += 1
            elif nums[i] == nums[i + 1]:
                return False
        return nums[4] - nums[k] < 5
```

# [剑指 Offer 55 - II. 平衡二叉树](https://leetcode-cn.com/problems/ping-heng-er-cha-shu-lcof/)

输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

 

示例 1:
```
给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7
返回 true 。
```
示例 2:
```
给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4
返回 false 。
```
 

限制：
```
0 <= 树的结点个数 <= 10000
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
    def isBalanced(self, root: TreeNode) -> bool:
        # 解法一：
        def recur(root: TreeNode) -> int:
            if root is None:
                return 0
            
            depleft = recur(root.left)
            if depleft == -1:
                return -1

            depright = recur(root.right)
            if depright == -1:
                return -1
            return max(depleft, depright) + 1 if abs(depleft - depright) <= 1 else -1
        
        return recur(root) != -1


class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        # 解法二：
        if root is None:
            return True
        
        return abs(self.depth(root.left) - self.depth(root.right)) <= 1 and \
                self.isBalanced(root.left) and self.isBalanced(root.right)

    def depth(self, root: TreeNode) -> int:
        if root is None:
            return 0
        return max(self.depth(root.left), self.depth(root.right)) + 1
```

# [剑指 Offer 62. 圆圈中最后剩下的数字](https://leetcode-cn.com/problems/yuan-quan-zhong-zui-hou-sheng-xia-de-shu-zi-lcof/)

0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字（删除后从下一个数字开始计数）。求出这个圆圈里剩下的最后一个数字。

例如，0、1、2、3、4这5个数字组成一个圆圈，从数字0开始每次删除第3个数字，则删除的前4个数字依次是2、0、4、1，因此最后剩下的数字是3。

 

示例 1：
```
输入: n = 5, m = 3
输出: 3
```
示例 2：
```
输入: n = 10, m = 17
输出: 2
```

限制：
```
1 <= n <= 10^5
1 <= m <= 10^6
```

代码：
```python3
class Solution:
    def lastRemaining(self, n: int, m: int) -> int:
        ans = 0
        for i in range(2, n + 1):
            ans = (ans + m) % i
        return ans
```

# [剑指 Offer 68 - I. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-zui-jin-gong-gong-zu-xian-lcof/)

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]

 <img src=https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/12/14/binarysearchtree_improved.png width="700">

示例 1:
```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。
```
示例 2:
```
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

说明:
```
所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉搜索树中。
```

代码：
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 迭代：
        if p.val > q.val: p, q = q, p
        while root:
            if root.val < p.val:
                root = root.right
            elif root.val > q.val:
                root = root.left
            else:
                break
        return root

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 递归：
        if root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        
        elif root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)

        else:
            return root
```

# [剑指 Offer 68 - II. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/er-cha-shu-de-zui-jin-gong-gong-zu-xian-lcof/)
此题详情与解法请见[腾讯精选练习50题236](https://github.com/ykaitao/leetcode_python/blob/master/%E8%85%BE%E8%AE%AF%E7%B2%BE%E9%80%89%E7%BB%83%E4%B9%A050%E9%A2%98-%E4%B8%AD%E7%AD%89.md#236-%E4%BA%8C%E5%8F%89%E6%A0%91%E7%9A%84%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88)
