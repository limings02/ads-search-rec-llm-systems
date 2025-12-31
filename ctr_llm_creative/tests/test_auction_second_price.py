"""Test second-price auction mechanics."""

import pytest
from contracts import AuctionStream, AuctionRecord, BidEvent, AuctionType


def test_auction_record_creation():
    """Test creating an auction record."""
    auction = AuctionRecord(
        auction_id="auction_001",
        timestamp=1000000,
        num_slots=1,
        auction_type=AuctionType.SECOND_PRICE,
    )
    
    assert auction.auction_id == "auction_001"
    assert auction.num_slots == 1
    assert auction.auction_type == AuctionType.SECOND_PRICE


def test_bid_event_creation():
    """Test creating a bid event."""
    bid = BidEvent(
        auction_id="auction_001",
        bidder_id="bidder_001",
        bid_price=5.0,
        advertiser_id="advertiser_001",
        winning=False,
    )
    
    assert bid.bid_price == 5.0
    assert bid.bidder_id == "bidder_001"


def test_auction_stream():
    """Test auction stream."""
    stream = AuctionStream(
        name="test_stream",
        version="1.0",
    )
    
    assert len(stream) == 0
    
    # Add an auction
    auction = AuctionRecord(
        auction_id="auction_001",
        timestamp=1000000,
    )
    stream.add_auction(auction)
    
    assert len(stream) == 1
    assert stream[0].auction_id == "auction_001"


def test_auction_stream_jsonl():
    """Test auction stream JSONL serialization."""
    stream = AuctionStream(
        name="test_stream",
        version="1.0",
    )
    
    auction = AuctionRecord(
        auction_id="auction_001",
        timestamp=1000000,
        bids=[
            BidEvent(
                auction_id="auction_001",
                bidder_id="bidder_001",
                bid_price=5.0,
                advertiser_id="advertiser_001",
            )
        ],
    )
    stream.add_auction(auction)
    
    # Serialize
    jsonl = stream.to_jsonl()
    assert "auction_001" in jsonl
    
    # Deserialize
    loaded_stream = AuctionStream.from_jsonl(jsonl, "test_stream", "1.0")
    assert len(loaded_stream) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
