# Android界面

## 基本知识

1 创建Fragment所需要参数.

创建一个Bundle对象,其参数使用putT相关方法写入,然后调用Fragment的setArguments关联这个Bundle对象.

2 启用Activity及所需参数.

创建一个Intent对象并给相应的Context/Activity.class,其参数使用putT相关方法写入,然后调用Context的startActivity启动Activity对象.

## RecyclerView界面开发

[RecyclerView的基本使用](https://www.jianshu.com/p/7a7d9301b2f1)

1 RecyclerView.ViewHolder表示一项item.

2 RecyclerView.Adapter表示items,item集合.

其RecyclerView的adapter/layoutManager对应数据与布局.

其中RecyclerView.Adapter中二个主要函数.

1 onCreateViewHolder 每个子view创建.

2 onBindViewHolder 绑定子view上的数据,并可以给相应控件绑定事件等.

注意: 现getItemViewType可以转化position成viewtype.其position暂时理解成添加到RecyclerView的view对象的索引.

RecyclerView 通过setAdapter,绑定与RecyclerView.Adapter关系.

## ViewPager View/Fragment之间切换

[ViewPager使用详解(二)](https://www.jianshu.com/p/d86e31dcc97b)

FragmentStatePagerAdapter/FragmentPagerAdapter: 其getItem(int position)返回具体的fragment.

TabLayout通过setupWithViewPager与ViewPager关联起来.

## DrawerLayout 侧滑区域

[DrawerLayout使用详解](https://blog.csdn.net/yechaoa/article/details/91452474)

Toolbar左边的三道杠,使用关联ActionBarDrawerToggle.

使用NavigationView块里的android:layout_gravity用来表明滑出的弹出块并取名.

## ButterKnife

[Butterknife](https://www.jianshu.com/p/ac6ee4760385)

ButterKnife是一个专注于Android系统的View注入框架,以前总是要写很多findViewById来找到View对象，有了ButterKnife可以很轻松的省去这些步骤。
