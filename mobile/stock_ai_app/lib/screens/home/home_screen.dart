import 'package:flutter/material.dart';

import '../ranking/ranking_screen.dart';
import '../backtest/backtest_screen.dart';
import '../settings/settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _selectedIndex = 0;

  final List<Widget> _pages = const [
    Center(
      child: Text(
        "ホーム",
        style: TextStyle(
          fontSize: 28,
          fontWeight: FontWeight.bold,
        ),
      ),
    ),
    RankingScreen(),
    BacktestScreen(),
    SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Stock AI"),
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        type: BottomNavigationBarType.fixed,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: "ホーム",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart),
            label: "ランキング",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.show_chart),
            label: "バックテスト",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.settings),
            label: "設定",
          ),
        ],
      ),
    );
  }
}