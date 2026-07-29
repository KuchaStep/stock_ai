import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../providers/ranking_provider.dart';

class RankingScreen extends ConsumerWidget {
  const RankingScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ranking = ref.watch(rankingProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text("AIランキング"),
      ),
      body: ranking.when(
        loading: () => const Center(
          child: CircularProgressIndicator(),
        ),

        error: (error, stackTrace) => Center(
          child: Text(error.toString()),
        ),

        data: (stocks) {
          return ListView.builder(
            itemCount: stocks.length,
            itemBuilder: (context, index) {
              final stock = stocks[index];

              return ListTile(
                leading: CircleAvatar(
                  child: Text("${index + 1}"),
                ),
                title: Text(stock.name),
                subtitle: Text(stock.code),
                trailing: Text(
                  "${stock.upProbability.toStringAsFixed(1)} %",
                  style: const TextStyle(
                    color: Colors.red,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              );
            },
          );
        },
      ),
    );
  }
}