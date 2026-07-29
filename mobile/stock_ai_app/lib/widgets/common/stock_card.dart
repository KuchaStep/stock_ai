import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../models/ranking_model.dart';

class StockCard extends StatelessWidget {
  final RankingModel stock;
  final int rank;
  final VoidCallback? onTap;

  const StockCard({
    super.key,
    required this.stock,
    required this.rank,
    this.onTap,
  });

  String getRankLabel() {
    switch (rank) {
      case 1:
        return "🥇";
      case 2:
        return "🥈";
      case 3:
        return "🥉";
      default:
        return rank.toString();
    }
  }

  @override
  Widget build(BuildContext context) {
    final formatter = NumberFormat("#,###");

    return Card(
      elevation: 2,
      margin: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 8,
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              Row(
                children: [
                  Text(
                    getRankLabel(),
                    style: const TextStyle(fontSize: 24),
                  ),
                  const SizedBox(width: 12),

                  Expanded(
                    child: Text(
                      stock.name,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),

                  Text(
                    "${stock.upProbability.toStringAsFixed(1)}%",
                    style: const TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              Row(
                children: [
                  Text(
                    stock.code,
                    style: TextStyle(
                      color: Colors.grey[600],
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text("現在値"),
                  Text(
                    "¥${formatter.format(stock.close)}",
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 8),

              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text("更新日"),
                  Text(
                    DateFormat("yyyy/MM/dd").format(stock.date),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}