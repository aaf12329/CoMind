# JavaScript 入门笔记

> 从已有的 C / Python 经验出发，先掌握能直接使用的语法，再补充容易踩坑的规则。
>
> 代码风格：**4 空格缩进，左右大括号各占一行；创建自定义对象优先使用 `class`。**

[返回前端学习索引](README.md) · [运行工具](语言运行与工程工具.md) · [学习清单](JavaScript科技树_学习清单.md)

## 目录

1. [文件组织与学习约定](#1-文件组织与学习约定)
2. [输出：控制台与弹窗](#2-输出控制台与弹窗)
3. [变量：let 与 const](#3-变量let-与-const)
4. [类：class、constructor 与 this](#4-类classconstructor-与-this)
5. [bind：固定函数中的 this](#5-bind固定函数中的-this)
6. [箭头函数：另一种函数写法](#6-箭头函数另一种函数写法)
7. [快速回顾](#7-快速回顾)

---

## 1. 文件组织与学习约定

### HTML、JavaScript、CSS 分开存放

这是当前的项目组织约定，方便管理和维护。

```text
项目文件夹/
├── index.html    页面结构
├── main.js       JavaScript 逻辑
└── style.css     页面样式
```

在 HTML 中引入独立的 JavaScript 文件：

```html
<script src="./main.js" defer></script>
```

`src` 指定脚本路径；这里的 `defer` 让外部脚本在 HTML 解析完成后执行。

在 HTML 中引入独立的 CSS 文件：

```html
<link rel="stylesheet" href="./style.css">
```

`href` 指定样式文件路径，`rel="stylesheet"` 表示引入样式表。

### 当前采用的约定

| 项目 | 当前做法 |
| --- | --- |
| JavaScript | 写在独立的 `.js` 文件中，由 HTML 引入。 |
| CSS | 写在独立的 `.css` 文件中，由 HTML 引入。 |
| 第三方依赖 | 优先通过 npm 安装和管理，暂不采用 CDN 引入方式。 |
| 变量声明 | 使用 `let` 和 `const`，当前学习代码不使用 `var`。 |
| 自定义对象 | 优先使用 `class` 创建。 |
| jQuery | 暂不纳入当前学习路线。 |

> **补充：** npm 安装依赖与浏览器加载依赖是两步。包安装好后，如何引入和运行，要看项目采用的工具与包的用法；不能认为执行一次 `npm install` 后，浏览器就能直接使用所有包。
>
> `var` 和 jQuery 仍然存在；这里记录的是个人学习选择。

---

## 2. 输出：控制台与弹窗

| 写法 | 作用 | 查看位置 |
| --- | --- | --- |
| `console.log(...)` | 打印信息，便于观察变量和调试。 | 浏览器开发者工具的 Console，或 Node.js 运行所在的终端。 |
| `alert(...)` | 弹出提示框。 | 浏览器页面。 |

```javascript
console.log("程序开始运行");

alert("这是一个提示框");
```

> `alert` 是浏览器提供的接口，普通 Node.js 脚本中默认没有这个函数。弹窗出现后，需要先关闭它，当前脚本才会继续往下执行。

---

## 3. 变量：let 与 const

### 核心区别

| 对比项 | `let` | `const` |
| --- | --- | --- |
| 声明时必须赋初值吗？ | 不必；不赋值时为 `undefined`。 | 必须。 |
| 能重新赋值吗？ | 可以。 | 不可以。 |
| 能保存不同类型的值吗？ | 可以。 | 可以；但初始化后不能换成另一个值。 |
| 作用域 | 块作用域。 | 块作用域。 |

```javascript
let count;
console.log(count); // undefined

count = 10;
count = 20; // 可以重新赋值。

const limit = 100;
console.log(limit);

// limit = 200; // 不允许：const 不能重新赋值。
```

**使用习惯：** 不需要重新赋值时使用 `const`，需要重新赋值时使用 `let`。

### 块作用域

`let` 和 `const` 声明的变量受所在 `{ }` 代码块限制。可以借用 C 的块作用域经验理解，但两种语言的全部作用域规则并不相同。

```javascript
if (true)
{
    let count = 10;
    console.log(count); // 可以访问。
}

// console.log(count); // 这里无法访问上面的 count。
```

### 补充：const 不会冻结对象

初学时先记住一句话：

> **`const` 限制的是变量重新赋值，不是对象内部内容的修改。**

如果变量保存的是对象或数组，可以修改其中允许修改的属性或元素；能否修改还取决于对象本身的限制。

后面的 `person` 示例会展示：变量始终指向同一个实例，但实例的姓名和年龄可以改变。

---

## 4. 类：class、constructor 与 this

### 定义一个带默认属性的类

```javascript
class Person
{
    name = "";
    age = 0;
}
```

这里定义了两个实例属性：`name` 的初始值是空字符串，`age` 的初始值是 `0`。

### 使用 constructor 初始化

下面是另一种定义方式，通过构造函数接收初始值：

```javascript
class Person
{
    constructor(name, age)
    {
        this.name = name;
        this.age = age;
    }
}

const person = new Person("张三", 18);

console.log(person.name); // 张三
console.log(person.age);  // 18
```

> 本节两个 `Person` 类是替代示例，分别阅读或运行，不要在同一作用域内重复声明同名类。

### 拆开看创建语句

```javascript
const person = new Person("张三", 18);
```

| 部分 | 含义 |
| --- | --- |
| `Person` | 类名。 |
| `new` | 创建类的实例。 |
| `"张三", 18` | 传给构造函数的参数。 |
| `constructor(name, age)` | 创建实例时执行的初始化逻辑。 |
| `this` | 在这里指向正在创建的实例。 |
| `person` | 保存实例引用的变量。 |

### 与 Python 的经验对应

JavaScript 的 `constructor` 可以类比 Python 的 `__init__`，用于初始化实例。这里是用途上的类比，具体机制并不完全相同。

- JavaScript 不需要在参数列表中手动写 `self`。
- 在构造函数内部，通过 `this.name`、`this.age` 访问实例属性。
- 没有特殊初始化逻辑时，可以不显式编写 `constructor`。

---

## 5. bind：固定函数中的 this

> **进阶补充：** 扫基础时可以先跳过。以后需要把对象方法单独取出来调用，再回来查看。

### 它解决什么问题？

普通函数中的 `this` 通常与调用方式有关。把方法从对象上取出来后，直接调用，原来的对象关系可能丢失。

```javascript
const second = person.introduce.bind(person);
```

这句代码的含义是：**创建一个新函数，让它调用 `introduce` 时，`this` 固定指向 `person`。**

- 调用 `bind(person)` 时，不会立刻执行 `introduce`。
- 后续调用 `second()`，才会执行对应的方法。
- 绑定的是实例引用，并没有复制或冻结实例当时的数据。

### 完整示例

```javascript
class Person
{
    constructor(name, age)
    {
        this.name = name;
        this.age = age;
    }

    introduce()
    {
        console.log(`我叫${this.name}，今年${this.age}岁。`);
    }
}

// 创建实例，自动执行 constructor。
const person = new Person("张三", 18);

// 通过实例调用，this 指向 person。
person.introduce();
// 输出：我叫张三，今年18岁。

// 创建绑定函数，此时不会执行 introduce。
const second = person.introduce.bind(person);

// 调用绑定函数，this 固定指向 person。
second();
// 输出：我叫张三，今年18岁。

// 修改同一个实例的数据，const 允许这种操作。
person.name = "李四";
person.age = 20;

// 再次调用，读取同一实例修改后的数据。
second();
// 输出：我叫李四，今年20岁。

// person = new Person("王五", 22);
// 不允许：这会给 const 变量重新赋值。
```

**记忆方式：** `bind` 固定访问哪个对象，之后读取的仍然是该对象当前的数据。

---

## 6. 箭头函数：另一种函数写法

在只接收参数、处理数据并返回结果的常见场景中，可以先把箭头函数当作函数的另一种写法。

### 对照普通函数表达式

```javascript
const addNormal = function(a, b)
{
    return a + b;
};

const addArrow = (a, b) =>
{
    return a + b;
};

console.log(addNormal(2, 3)); // 5
console.log(addArrow(2, 3));  // 5
```

两者都能接收参数，也都能返回结果，调用方式相同。

### 简写：表达式结果直接返回

当函数体只有一个要返回的表达式时，可以省略 `{ }` 和 `return`：

```javascript
const add = (a, b) => a + b;
```

如果写了 `{ }`，需要显式使用 `return` 才能返回对应的结果：

```javascript
const add = (a, b) =>
{
    return a + b;
};
```

### 与普通函数不完全相同

| 特性 | 普通函数 | 箭头函数 |
| --- | --- | --- |
| `this` | 有自己的绑定规则，通常取决于调用方式。 | 没有自己的 `this`，沿用定义处外层的 `this`。 |
| `arguments` | 有自己的实参集合。 | 没有自己的 `arguments`。 |
| 作为构造函数被 `new` 调用 | 某些普通函数可以；类方法等不能。 | 不可以。 |

这些区别在任何位置都存在。类和对象的方法中容易看到 `this` 差异，嵌套函数、定时器和事件回调中也可能遇到。

**选择时看函数需要哪些特性，而不是只看它有没有写在类或对象里。**

---

## 7. 快速回顾

| 想做什么 | 优先记住的写法 |
| --- | --- |
| 声明之后需要重新赋值的变量 | `let count = 0;` |
| 声明之后不需要重新赋值的变量 | `const limit = 100;` |
| 打印调试信息 | `console.log(value);` |
| 在浏览器中弹出提示 | `alert("提示内容");` |
| 定义类 | `class Person` |
| 初始化实例 | `constructor(name, age)` |
| 创建实例 | `new Person("张三", 18)` |
| 在构造函数中访问当前实例 | `this.name` |
| 固定普通函数的 `this` | `method.bind(person)` |
| 编写简单计算函数 | `(a, b) => a + b` |

*根据原 JS.txt 整理，并补入近期关于箭头函数的讨论。更新：2026-10-18。*

*2026-10-25：由桌面 JS.md 归档；保留基础示例，和运行工具、学习路线交叉引用。示例用于讲语法，不自动声明符合可选临时草案的全部条款。*
